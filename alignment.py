import re
import sys
from itertools import repeat
import textwrap
import shutil

from nltk.metrics.distance import edit_distance
import numpy as np

punct_tokens = (",", ":", ".", ";", "!", "?", "¿", "¡")
punct_re = re.compile("[-;:,.!?¡¿()\"]", re.UNICODE)


def word_tokenize(text, strip_punctuation=True):
    if strip_punctuation:
        tokens = re.findall(r"[\w']+|[.,!?;:¿¡]", text)
        word_tokens = [token for token in tokens if token not in punct_tokens]
    else:
        word_tokens = text.split()
    return word_tokens


def print_alignment(alignment):
    # standalone function so alignment objects don't have to worry about formatting
    # but a similar case could be made for the make_string() method...
    columns, rows = shutil.get_terminal_size()
    width = min(len(alignment.aligned_query_string), columns - 6)
    wrapped_ref_lines = textwrap.wrap(alignment.aligned_ref_string, width=width)
    display_strings = []
    start_index = 0
    end_index = 0
    for i, ref_line in enumerate(wrapped_ref_lines):
        line_length = len(ref_line)
        end_index += line_length
        query_line = alignment.aligned_query_string[start_index:end_index + 1]
        alignment_line = alignment.alignment_string[start_index:end_index + 1]
        #count leading spaces, take them off query and alignment lines
        stripped_query_line = query_line.lstrip()
        leading_whitespace = query_line[:len(query_line) - len(stripped_query_line)]
        n_leading_spaces = len(leading_whitespace)
        stripped_alignment_line = alignment_line[n_leading_spaces:]
        display_strings.append("\nSeq1: {}".format(ref_line))
        if "|" in stripped_alignment_line:
            display_strings.append("     {}".format(stripped_alignment_line))
        display_strings.append("Seq2: {}".format(stripped_query_line))
        start_index += line_length
        start_index += n_leading_spaces
        end_index += n_leading_spaces
    print("\n".join(display_strings), '\n')



def char_distance(char_1, char_2):
    # case_re = re.compile(char_1, re.IGNORECASE)
    return 0 if char_1 == char_2 else 2


def same_ignoring_case_and_punctuation(word_1, word_2):
    # is there a faster way?
    w_1 = re.sub(punct_re, "", word_1).lower()
    w_2 = re.sub(punct_re, "", word_2).lower()
    if w_1 == w_2:
        return True
    return False


def scaled_edit_distance(word_1, word_2):
    edit_dist = edit_distance(word_1, word_2)
    max_len = max(len(word_1), len(word_2))
    return 2 * (edit_dist / float(max_len))


def word_distance(word_1, word_2):
    # has trouble with parens
    if word_1 == word_2:
        return 0
    #elif same_ignoring_case_and_punctuation(word_1, word_2):
    #    return 0
    # Removing this was probably a premature optimization.
    else:
        return scaled_edit_distance(word_1, word_2)

        #too slow and doesn't benefit us much?
        #aligned_seq1, alignment, aligned_ref, perc_correct = self.affine_global_align(word_1, word_2)
        #return -1 - ((100 - perc_correct) / 50)
        #return -2


class Alignment(object):
    def __init__(self, ref_sequence, query_sequence, ref_to_align=None, query_to_align=None, gap_extend=0.5, gap_open=2.5):
        self.gap_extend = gap_extend
        self.gap_open = gap_open
        self.ref_sequence = ref_sequence
        self.query_sequence = query_sequence
        self.ref_to_align = ref_sequence if ref_to_align is None else ref_to_align
        self.query_to_align = query_sequence if query_to_align is None else query_to_align
        self.alignment, self.aligned_ref, self.aligned_query = ([],) * 3
        self.aligned_ref_string, self.alignment_string, self.aligned_query_string = ("",) * 3
        self.score = None
        self.affine_global_align()
        self.make_strings()


    def affine_global_align(self):
        query_to_align = self.query_to_align
        ref_to_align = self.ref_to_align
        query = self.query_sequence
        ref = self.ref_sequence

        """
        For affine:
        create three alignment score matrices
            left
            down
            diag
        and have free edges from down or left to corresponding diag
        and (gap_open + gap_extend) edges from diag to one down
        and (gap_open + gap_extend) edges from diag to one left
        """

        n_rows = len(query_to_align) + 1
        n_cols = len(ref_to_align) + 1

        vertical_matrix = np.zeros((n_rows, n_cols))
        horizontal_matrix = np.zeros((n_rows, n_cols))
        diagonal_matrix = np.zeros((n_rows, n_cols))


        for i in range(0, n_rows):
            cost = i * self.gap_open
            vertical_matrix[i][0] = cost # change these to not penalize terminal gaps, or at least not as heavily?
            horizontal_matrix[i][0] = cost

        for j in range(0, n_cols):
            cost = j * self.gap_open
            vertical_matrix[0][j] = cost
            horizontal_matrix[0][j] = cost

        # Can I avoid filling in all entries in this matrix?
        for i in range(1, n_rows):
            for j in range(1, n_cols):
                vertical_extend = vertical_matrix[i - 1][j] + self.gap_extend
                vertical_open = diagonal_matrix[i - 1][j] + self.gap_open + self.gap_extend
                vertical_matrix[i][j] = min(vertical_extend, vertical_open)

                horizontal_extend = horizontal_matrix[i][j - 1] + self.gap_extend
                horizontal_open = diagonal_matrix[i][j - 1] + self.gap_open + self.gap_extend
                horizontal_matrix[i][j] = min(horizontal_extend, horizontal_open)

                match = diagonal_matrix[i - 1][j - 1] + self.distance(query_to_align[i - 1], ref_to_align[j - 1])
                vertical = vertical_matrix[i][j]
                horizontal = horizontal_matrix[i][j]
                diagonal_matrix[i][j] = min(match, vertical, horizontal)

        """
        Track back to create the aligned sequence pair
        If in diagonal: can either match, go to the vertical (open gap), or go to the horizontal (open gap)
        If in vertical: can either continue vertical (extend gap) or return to diagonal (free, no index change)
        If in horizontal: can either continue horizontal (extend gap) or return to diagonal (free, no index change)
        """
        alignment = []
        aligned_ref = []
        aligned_query = []
        i = len(query)
        j = len(ref)
        matrices = {"diagonal": diagonal_matrix, "vertical": vertical_matrix, "horizontal": horizontal_matrix}
        current_matrix_key = "diagonal"

        total_score = 0
        while i > 0 and j > 0:
            #print(i, j)
            current_matrix = matrices[current_matrix_key]
            score = current_matrix[i][j]
            total_score += score
            element_1 = query_to_align[i - 1]
            element_2 = ref_to_align[j - 1]
            orig_element_1 = query[i - 1]
            orig_element_2 = ref[j - 1]

            if current_matrix_key == "diagonal":
                diag_score = diagonal_matrix[i - 1][j - 1]
                up_score = vertical_matrix[i][j]
                left_score = horizontal_matrix[i][j]
                distance = self.distance(element_1, element_2)
                if score == diag_score + distance:
                    if distance == 0:
                        alignment.append("match")
                    else:
                        alignment.append("mismatch")
                    aligned_query.append(orig_element_1)
                    aligned_ref.append(orig_element_2)
                    i -= 1
                    j -= 1
                elif score == left_score:
                    current_matrix_key = "horizontal"
                elif score == up_score:
                    current_matrix_key = "vertical"
                else:
                    sys.stderr.write('Not Possible')
            elif current_matrix_key == "vertical":
                diag_score = diagonal_matrix[i - 1][j]
                if score == diag_score + self.gap_open + self.gap_extend:
                    current_matrix_key = "diagonal"
                aligned_query.append(orig_element_1)
                aligned_ref.append("")
                alignment.append("gap")
                i -= 1
            elif current_matrix_key == "horizontal":
                diag_score = diagonal_matrix[i][j - 1]
                if score == diag_score + self.gap_open + self.gap_extend:
                    current_matrix_key = "diagonal"
                aligned_query.append("")
                aligned_ref.append(orig_element_2)
                alignment.append("gap")
                j -= 1
        while i > 0:
            #print(i, j)
            orig_element_1 = query[i - 1]
            aligned_query.append(orig_element_1)
            aligned_ref.append("")
            alignment.append("gap")
            total_score += self.gap_extend
            i -= 1
        while j > 0:
            #print(i, j)
            orig_element_2 = ref[j - 1]
            aligned_query.append("")
            aligned_ref.append(orig_element_2)
            alignment.append("gap")
            total_score += self.gap_extend
            j -= 1

        self.alignment = list(reversed(alignment))
        self.aligned_ref = list(reversed(aligned_ref))
        self.aligned_query = list(reversed(aligned_query))
        self.score = total_score / float(len(alignment))

    def reference_scores(self):
        ref_length = len(self.ref_sequence)
        scores = [0] * ref_length
        ref_index = 0
        alignment_index = 0
        in_ref_gap = False
        before_ref_gaps = []
        after_ref_gaps = []
        while ref_index < ref_length:
            align_item = self.alignment[alignment_index]
            if align_item == 'gap':
                query_item = self.aligned_query[alignment_index]
                if query_item == "":
                    ref_index += 1
                else:
                    if not in_ref_gap:
                        in_ref_gap = True
                        if ref_index > 0:
                            before_ref_gaps.append(ref_index - 1)
            else:
                if in_ref_gap:
                    after_ref_gaps.append(ref_index)
                    in_ref_gap = False
                if align_item == 'match':
                    scores[ref_index] = 1
                    ref_index += 1
                elif align_item == 'mismatch':
                    ref_index += 1

            alignment_index += 1

        for index in before_ref_gaps + after_ref_gaps:
            scores[index] = 0
        return scores


class CharAlignment(Alignment):
    def __init__(self, ref_string, query_string, gap_extend=0.5, gap_open=2.5, case_sensitive=False):
        self.distance = char_distance
        ref_to_align = ref_string if case_sensitive else ref_string.lower()
        query_to_align = query_string if case_sensitive else query_string.lower()
        super(self.__class__, self).__init__(ref_string, query_string, ref_to_align=ref_to_align,
                                             query_to_align=query_to_align, gap_extend=gap_extend,
                                             gap_open=gap_open)

    def make_strings(self):
        self.aligned_query_string = "".join(["_" if ch == "" else ch for ch in self.aligned_query])
        self.aligned_ref_string = "".join(["_" if ch == "" else ch for ch in self.aligned_ref])
        self.alignment_string = "".join([" " if rel == "match" else "|" for rel in self.alignment])


class WordAlignment(Alignment):
    def __init__(self, ref_string, query_string, gap_extend=0.5, gap_open=2.5, punctuation_sensitive=False, case_sensitive=False):
        self.distance = word_distance
        ref_sequence = word_tokenize(ref_string, strip_punctuation=False)
        query_sequence = word_tokenize(query_string, strip_punctuation=False)
        ref, query = (ref_string, query_string) if case_sensitive else (ref_string.lower(), query_string.lower())
        ref_to_align, query_to_align = [word_tokenize(string, strip_punctuation=not punctuation_sensitive) for string in [ref, query]]
        super(self.__class__, self).__init__(ref_sequence, query_sequence,
                                             ref_to_align=ref_to_align, query_to_align=query_to_align,
                                             gap_extend=gap_extend, gap_open=gap_open)

    def make_strings(self):
        ref_strings = []
        query_strings = []
        alignment_strings = []
        for i in range(len(self.alignment)):
            align_item = self.alignment[i]
            ref_item = self.aligned_ref[i]
            query_item = self.aligned_query[i]
            if align_item == 'gap':
                if ref_item == "":
                    gap_len_item = query_item
                    string_set_of_gap = ref_strings
                    string_set_of_nongap = query_strings
                else:
                    gap_len_item = ref_item
                    string_set_of_gap = query_strings
                    string_set_of_nongap = ref_strings
                gap_string = "".join(repeat("_", len(gap_len_item)))
                string_set_of_gap.append(gap_string)
                string_set_of_nongap.append(gap_len_item)
                align_string = "".join(repeat("|", len(gap_string)))
            else:
                max_len = max(len(query_item), len(ref_item))
                query_filler = "".join(repeat("_", max_len - len(query_item)))
                ref_filler = "".join(repeat("_", max_len - len(ref_item)))

                if align_item == 'match':
                    align_char = ' '
                    query_strings.append(query_item + query_filler)
                    ref_strings.append(ref_item + ref_filler)
                    align_string = "".join(repeat(align_char, max_len))
                else:
                    edit_dist = scaled_edit_distance(ref_item, query_item)
                    if edit_dist < 1:
                        ca = CharAlignment(ref_item, query_item)
                        query_strings.append(ca.aligned_query_string)
                        ref_strings.append(ca.aligned_ref_string)
                        align_string = ca.alignment_string
                    else:
                        query_strings.append(query_item + query_filler)
                        ref_strings.append(ref_item + ref_filler)
                        align_string = "".join(repeat('|', max_len))
            alignment_strings.append(align_string)
        self.aligned_query_string = " ".join(query_strings)
        self.aligned_ref_string = " ".join(ref_strings)
        self.alignment_string = " ".join(alignment_strings)
