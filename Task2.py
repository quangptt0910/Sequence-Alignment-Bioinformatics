import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
"""
    Quantitative matching of sequence pairs
Prepare a properly functioning program for quantitative comparison of pairs of protein or DNA
coding sequences using the Needleman-Wunsch global matching algorithm
The program should control the correctness of the input data, generate correct results, as well as be
concise, legible (with comments) and consistent with the style rules of the programming language
used.
IMPORTANT NOTE:
sequence
matching point
SHOW how the match between2 sequences
"""

def load_sequence_manual():
    """
    Prompt the user to enter two sequences
    :return: sequence1, sequence2
    """
    seq1 = input("Please enter two sequences: ").strip().upper()
    seq2 = input("Please enter two sequences: ").strip().upper()
    return seq1, seq2

def load_sequence_fasta(path1, path2):
    """
    Load one sequence from each of two FASTA files.

    Args:
        path1 (str): path to first FASTA file (one record)
        path2 (str): path to second FASTA file (one record)
    Returns: seq1, seq2 (str)
    """
    def _read_single_fasta(path):
        seq = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('>'):
                    continue
                seq.append(line)
        if not seq:
            print(f"No sequence found in {path}")
            sys.exit(1)
        return ''.join(seq)

    seq1 = _read_single_fasta(path1)
    seq2 = _read_single_fasta(path2)
    return seq1, seq2

def dotplot(seq1, seq2, window_size=1, threshold=1):
    """
     Dot plot table from sequence 1 and sequence 2
    :param seq1: first sequence
    :param seq2: second sequence
    :param window_size: size of window. Default is 1
    :param threshold: threshold for dot plot. Default is 1
    :return: the dot plot table - 1 when they have the same character, otherwise 0
    """
    len1 = len(seq1)
    len2 = len(seq2)
    # Initialize dotplot matrix with zeros
    matrix = np.zeros((len1, len2))

    # Case when window size is 1
    if window_size == 1:
        for i in range(len1):
            for j in range(len2):
                if seq1[i] == seq2[j]:
                    matrix[i][j] = 1

    else:
        for i in range(len1 - window_size + 1):
            for j in range(len2 - window_size + 1):
                # Compute the window around position i in seq1 and j in seq2
                matches = 0
                for k in range(window_size):
                    if i+k < len1 and j+k < len2 and seq1[i+k] == seq2[j+k]:
                        matches += 1

                if matches >= threshold:
                    matrix[i][j] = 1
                # windowA = seq1[max(i-window_size,0): min(i + window_size, len1)]
                # windowB = seq2[max(j-window_size,0): min(j + window_size, len2)]
                #
                # # Count the number of matching symbols
                # match = sum([1 for x, y in zip(windowA, windowB) if x == y])
                # if match >= threshold:
                #     matrix[i][j] = 1

    return matrix

def dotplot2Ascii(dp, seq1, seq2, heading, filename):
    """
    Print the dot plot table from sequence 1 and sequence 2 to the text file
    :param dp: dot plot table
    :param seq1: sequence 1 (to take the first sequence name)
    :param seq2: sequence 2 (to take the second sequence name)
    :param heading: Heading name of the result file
    :param filename: filename to save the result
    :return: a (text) file of dot plot table with the heading and 2 sequences
    """
    try:
        # Open the output
        with open(filename, 'w') as f:
            # Write the heading
            f.write(heading + "\n")
            f.write(" " * 4 + " ".join(seq2) + "\n")

            for i in range(len(seq1)):
                f.write(seq1[i] + " | ")
                for j in range(len(seq2)):
                    if dp[i][j] == 1:
                        f.write("* ")
                    else:
                        f.write(". ")
                f.write("\n")
        print(f"Dot plot table saved to {filename}")
    except Exception as e:
        print(f"Error saving dot plot: {e}")

def dotplotGraphic(dp, labelA, labelB, heading):
    """
    https://medium.com/@anoopjohny2000/visualizing-sequence-similarity-with-dotplots-in-python-f5cf0ac8559f
    :param dp:
    :param labelA:
    :param labelB:
    :param heading:
    :return:
    """
    # create a new figure
    fig, ax = plt.subplots()

    # plot the dots using a scatter plot
    rows, cols = np.where(dp == 1)
    ax.scatter(cols, rows, marker='.', color='black')

    # set the labels and title
    ax.set_xlabel(labelB)
    ax.set_ylabel(labelA)
    ax.set_title(heading)

    # set the tick positions and labels
    xticks = np.arange(0.5, dp.shape[1], 1)
    yticks = np.arange(0.5, dp.shape[0], 1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(np.arange(1, dp.shape[1] + 1))
    ax.set_yticklabels(np.arange(1, dp.shape[0] + 1)[::-1])

    # save the figure to a file and display it on screen
    # plt.savefig(filename)
    plt.tight_layout()
    plt.show()


"""
https://www.researchgate.net/publication/338420139_A_scalable_multiple_pairwise_protein_sequence_alignment_acceleration_using_hybrid_CPU-GPU_approach
https://medium.com/@nandiniumbarkar/needleman-wunsch-algorithm-7bba68b510db
https://www.cs.sjsu.edu/~aid/cs152/NeedlemanWunsch.pdf
"""
def scoring_path(seq1, seq2, match=1, mismatch=0, gap=-1):
    """
    Calculate the scoring matrix for sequence alignment
    """
    len1, len2 = len(seq1), len(seq2)
    path = np.zeros((len1 + 1, len2 + 1), dtype=int)

    direction = np.zeros_like(path)
    # 0 diagonal, 1 Up, 2 left

    # first row and col as penalties
    for i in range(1, len1 + 1):
        path[i, 0] = i * gap
        direction[i, 0] = 1 # up direction

    for j in range(1, len2 + 1):
        path[0, j] = j * gap
        direction[0, j] = 2 # left direction

    # filling matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            diag_score = path[i - 1, j - 1] + (match if seq1[i - 1] == seq2[j - 1] else mismatch)
            up_score = path[i-1, j] + gap
            left_score = path[i, j - 1] + gap

            path[i, j] = max(up_score, left_score, diag_score)

            if path[i, j] == diag_score:
                direction[i, j] = 0
            elif path[i, j] == up_score:
                direction[i, j] = 1
            else:
                direction[i, j] = 2

    return path, direction

def traceback_alignment(seq1, seq2, direction):
    """
    Traceback through the direction matrix to find the optimal alignment
    :param seq1:
    :param seq2:
    :param direction:
    :return:
    """
    aligned_seq1 = []
    aligned_seq2 = []
    path_coord = [] # have the index of column/row

    i = len(seq1)
    j = len(seq2)
    path_coord.append((i, j))

    while i > 0 or j > 0:
        # Handle edge moves
        if i == 0: # only left move possible/ get to first row
            j -= 1
            aligned_seq1.append('-')
            aligned_seq2.append(seq2[j])
        elif j == 0: # Only up move possible
            i -= 1
            aligned_seq1.append(seq1[i])
            aligned_seq2.append('-')

        else:
            d = direction[i, j]
            if d == 0: # Diagonal
                i -= 1
                j -= 1
                aligned_seq1.append(seq1[i])
                aligned_seq2.append(seq2[j])
            elif d == 1: # Up
                i -= 1
                aligned_seq1.append(seq1[i])
                aligned_seq2.append('-')
            else: # Left
                j -= 1
                aligned_seq1.append('-')
                aligned_seq2.append(seq2[j])

        path_coord.append((i, j))


    # Reverse the coordinate path for the result
    return ''.join(aligned_seq1[::-1]), ''.join(aligned_seq2[::-1]), path_coord[::-1]

def visualize_alignment(seq1, seq2, score, path_coord):
    """
    Visualize the alignment scoring matrix and traceback path
    :param seq1:
    :param seq2:
    :param score:
    :param direction:
    :param path_coord:
    :return:
    """
    # Create figure and axis
    plt.figure(figsize=(len(seq1) + 2, len(seq2) + 2))
    ax = plt.gca() # Create new Axes using Figure

    # Get dimensions
    rows, cols = score.shape

    # for visualize matrix one larger in each dimension for sequences
    visual_rows = rows + 1
    visual_cols = cols + 1

    # Create a mask for the alignment path cells
    path_set = set((y + 1, x + 1) for y, x in path_coord)

    # Draw the grid and fill cells
    for i in range(visual_rows):
        for j in range(visual_cols):
            # Cell border
            rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='gray', linewidth=1)
            ax.add_patch(rect)

            # Top left
            if i == 0 and j == 0:
                ax.text(j + 0.5, i + 0.5, "D", ha='center', va='center',fontweight='bold')

            # First row will contain seq2
            elif i == 0 and j >= 2:
                if j - 2 < len(seq2):
                    ax.text(j + 0.5, i + 0.5,f"{seq2[j - 2]}{j - 1}", ha='center', va='center')

            elif i >= 2 and j == 0:
                if i - 2 < len(seq1):
                    ax.text(j + 0.5, i + 0.5,f"{seq1[i - 2]}{i - 1}", ha='center', va='center')

            # Add score value in each cell - contains other score from score matrix
            elif i >= 1 and j >= 1:

                score_i = i - 1
                score_j = j - 1
                if (i, j) in path_set:
                    rect_fill = plt.Rectangle((j, i), 1, 1, fill=True, facecolor='royalblue', alpha=0.5)
                    ax.add_patch(rect_fill)
                if score_i < rows and score_j < cols:
                    ax.text(j + 0.5, i + 0.5, f"{score[score_i, score_j]}",
                        ha='center', va='center', fontsize=11)

    # Calculate alignment score (value at bottom-right cell)
    alignment_score = score[rows - 1, cols - 1]
    ax.text(1, rows + 1.5, f"Score: {alignment_score}", ha='center', va='center',
            fontsize=12, fontweight='bold', color='gray')

    # Set limits and remove ticks
    ax.set_xlim(-0.1, visual_cols + 0.1)
    ax.set_ylim(visual_rows + 0.1, -0.1)  # Inverted y-axis to match matrix orientation
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove borders
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add title
    plt.title(f"Needleman-Wunsch Alignment Matrix", fontsize=12)

    plt.tight_layout()
    plt.show()


def run_needleman_wunsch_visualization(seq1, seq2, match=1, mismatch=-1, gap=-1):
    """
    Run the Needleman-Wunsch algorithm and visualize the results with the requested offset layout
    """
    # Calculate score matrix and direction matrix
    score_matrix, direction_matrix = scoring_path(seq1, seq2, match, mismatch, gap)

    # Get the traceback path
    aligned_seq1, aligned_seq2, path_coords = traceback_alignment(seq1, seq2, direction_matrix)

    # If your traceback function doesn't return path coordinates as expected:
    # path_coords = get_alignment_path(direction_matrix)

    # Visualize the alignment matrix
    visualize_alignment(seq1, seq2, score_matrix, path_coords)

    # Print the alignment
    print("Aligned Sequence 1:", aligned_seq1)
    print("                   ", ''.join(['|' if aligned_seq1[i] == aligned_seq2[i] and
                                                 aligned_seq1[i] != '-' and aligned_seq2[i] != '-'
                                          else ' ' for i in range(len(aligned_seq1))]))
    print("Aligned Sequence 2:", aligned_seq2)

    return aligned_seq1, aligned_seq2

if __name__ == "__main__":
    seq1 = "ACGCACTA"
    seq2 = "ACTGATTA"
    seq1x = "GCTGAGTGAGT"
    seq2x = "GCTAGTGTGT"
    window_size = 1
    threshold = 1
    #
    dp = dotplot(seq1, seq2, window_size, threshold)

    # Generate the ASCII dotplot
    dotplot2Ascii(dp, seq1, seq2, "Testing", "result.txt")

    # dotplotGraphic(dp, seq1, seq2, "sequence alignment example")
    score, direction = scoring_path(seq1, seq2)

    print(score)

    aligned1, aligned2, path = traceback_alignment(seq1, seq2, direction)

    print(aligned1)
    print(aligned2)
    print(path)

    visualize_alignment(seq1, seq2, score, path)

    run_needleman_wunsch_visualization(seq1, seq2)
