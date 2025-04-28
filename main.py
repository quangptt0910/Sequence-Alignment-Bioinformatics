import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib.patches as mpatches
from nw import *



def calculate_matching_statistics(aligned_seq1, aligned_seq2):
    """
    Calculate matching statistics for the alignment of two sequences.
    :param aligned_seq1:
    :param aligned_seq2:
    :return: Dictionary of matching statistics
    """
    total_length = len(aligned_seq1)
    matches = sum(1 for i in range(total_length) if aligned_seq1[i] == aligned_seq2[i] and
                  aligned_seq1 != '-' and aligned_seq2 != '-')

    gaps_seq1 = aligned_seq1.count('-')
    gaps_seq2 = aligned_seq2.count('-')

    total_gaps = gaps_seq1 + gaps_seq2

    non_gap_position = total_length - total_gaps
    percent_identity = (matches / non_gap_position) * 100 if non_gap_position > 0 else 0

    return {
        "length": total_length,
        "matches": matches,
        "gaps_seq1": gaps_seq1,
        "gaps_seq2": gaps_seq2,
        "total_gaps": total_gaps,
        "percent_identity": percent_identity,
    }

def display_alignment_result(aligned_seq1, aligned_seq2, statistics, parameters):
    """
    Display alignment result for given parameters.
    :param aligned_seq1:
    :param aligned_seq2:
    :param statistics: alignment statistics
    :param parameters: algorithm parameters
    """
    print("\n" + "=" * 60)
    print("ALIGNMENT RESULTS")
    print("=" * 60)

    print("\nAlgorithm Parameters:")
    print(f"Match score: {parameters['match']}")
    print(f"Mismatch score: {parameters['mismatch']}")
    print(f"Gap penalty: {parameters['gap']}")

    print("\nAlignment Statistics:")
    print(f"Alignment length: {statistics['length']}")
    print(f"Identical matches: {statistics['matches']}")
    print(f"Percent identity: {statistics['percent_identity']:.2f}%")
    print(f"Gaps in sequence 1: {statistics['gaps_seq1']}")
    print(f"Gaps in sequence 2: {statistics['gaps_seq2']}")
    print(f"Total gaps: {statistics['total_gaps']}")

    print("\nAlignment:")
    # Print in blocks of 60 characters for better readability
    for i in range(0, len(aligned_seq1), 60):
        block_seq1 = aligned_seq1[i:i + 60]
        block_seq2 = aligned_seq2[i:i + 60]

        # Create the middle line showing matches
        middle = ''
        for j in range(len(block_seq1)):
            if j < len(block_seq2) and block_seq1[j] == block_seq2[j] and block_seq1[j] != '-' and block_seq2[j] != '-':
                middle += '|'
            else:
                middle += ' '

        print(f"Seq1: {block_seq1}")
        print(f"      {middle}")
        print(f"Seq2: {block_seq2}")
        print()

def save_alignment_to_file(aligned_seq1, aligned_seq2, statistics, parameters, filename, extra_alignment=None):
    """
    Save alignment result to text file.
    :param aligned_seq1:
    :param aligned_seq2:
    :param statistics:
    :param parameters:
    :param filename:
    :param extra_alignment:
    """
    with open(filename, 'w') as f:
        f.write("NEEDLEMAN-WUNSCH SEQUENCE ALIGNMENT RESULTS\n")
        f.write("=" * 50 + "\n\n")

        f.write("Algorithm Parameters:\n")
        f.write(f"Match score: {parameters['match']}\n")
        f.write(f"Mismatch score: {parameters['mismatch']}\n")
        f.write(f"Gap penalty: {parameters['gap']}\n\n")

        f.write("Alignment Statistics:\n")
        f.write(f"Alignment length: {statistics['length']}\n")
        f.write(f"Identical matches: {statistics['matches']}\n")
        f.write(f"Percent identity: {statistics['percent_identity']:.2f}%\n")
        f.write(f"Gaps in sequence 1: {statistics['gaps_seq1']}\n")
        f.write(f"Gaps in sequence 2: {statistics['gaps_seq2']}\n")
        f.write(f"Total gaps: {statistics['total_gaps']}\n\n")

        f.write("Alignment:\n")
        # Print in blocks of 60 characters for prettier-print :))
        for i in range(0, len(aligned_seq1), 60):
            block_seq1 = aligned_seq1[i:i + 60]
            block_seq2 = aligned_seq2[i:i + 60]

            # Create the middle line showing matches
            middle = ''
            for j in range(len(block_seq1)):
                if j < len(block_seq2) and block_seq1[j] == block_seq2[j] and block_seq1[j] != '-' and block_seq2[
                    j] != '-':
                    middle += '|'
                else:
                    middle += ' '

            f.write(f"Seq1: {block_seq1}\n")
            f.write(f"      {middle}\n")
            f.write(f"Seq2: {block_seq2}\n\n")

        if extra_alignment:
            f.write(f"Found {len(extra_alignment)} optimal alignment path(s).\n\n")
            f.write("All Optimal Alignments:\n\n")
            for idx, (a1, a2) in enumerate(extra_alignment, 1):
                f.write(f"Optimal Alignment {idx}:\n")
                for i in range(0, len(a1), 60):
                    block_seq1 = a1[i:i + 60]
                    block_seq2 = a2[i:i + 60]

                    # Create the middle line showing matches
                    middle = ''
                    for j in range(len(block_seq1)):
                        if j < len(block_seq2) and block_seq1[j] == block_seq2[j] and block_seq1[j] != '-' and \
                                block_seq2[
                                    j] != '-':
                            middle += '|'
                        else:
                            middle += ' '

                f.write(f"Seq1: {a1}\n")
                f.write(f"      {''.join(['|' if x == y else ' ' for x, y in zip(a1, a2)])}\n")
                f.write(f"Seq2: {a2}\n\n")

    print(f"\nAlignment results saved to '{filename}'")


def find_all_optimal_paths(score_matrix, direction_matrix, seq1, seq2, i=None, j=None, path=None, all_paths=None):
    """
    Find all optimal alignment paths in the scoring matrix using backtracking

    Args:
        score_matrix: The scoring matrix
        direction_matrix: The direction matrix
        seq1: The first sequence
        seq2: The second sequence
        i, j: Current position in the matrix
        path: Current path being constructed
        all_paths: List to store all optimal paths

    Returns:
        list: All optimal alignment paths
    """
    if i is None and j is None:
        i, j = len(seq1), len(seq2)
        path = []
        all_paths = []

    path.append((i, j))

    if i == 0 and j == 0:
        # Reached the origin, this is a complete path
        all_paths.append(path.copy())
        path.pop()
        return all_paths

    # Check all possible previous cells that could have led to the current score
    current_score = score_matrix[i, j] if i > 0 and j > 0 else None

    # Check diagonal (match/mismatch)
    if i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            prev_score = score_matrix[i - 1, j - 1] + 1  # Match
        else:
            prev_score = score_matrix[i - 1, j - 1] - 1  # Mismatch

        if current_score == prev_score:
            find_all_optimal_paths(score_matrix, direction_matrix, seq1, seq2, i - 1, j - 1, path, all_paths)

    # Check up (gap in seq2)
    if i > 0:
        prev_score = score_matrix[i - 1, j] - 1  # Gap penalty
        if current_score == prev_score:
            find_all_optimal_paths(score_matrix, direction_matrix, seq1, seq2, i - 1, j, path, all_paths)

    # Check left (gap in seq1)
    if j > 0:
        prev_score = score_matrix[i, j - 1] - 1  # Gap penalty
        if current_score == prev_score:
            find_all_optimal_paths(score_matrix, direction_matrix, seq1, seq2, i, j - 1, path, all_paths)

    path.pop()  # Backtrack
    return all_paths


def visualize_all_optimal_paths(seq1, seq2, score_matrix, all_paths):
    """
    Visualize all optimal alignment paths on the scoring matrix

    Args:
        seq1: The first sequence
        seq2: The second sequence
        score_matrix: The scoring matrix
        all_paths: List of all optimal paths
    """
    plt.figure(figsize=(len(seq1) + 5, len(seq2) + 5))
    ax = plt.gca()

    # Get dimensions
    rows, cols = score_matrix.shape

    # for visualize matrix one larger in each dimension for sequences
    visual_rows = rows + 1
    visual_cols = cols + 1

    # Create a unique color for each path
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_paths)))

    # Draw the grid and fill cells
    for i in range(visual_rows):
        for j in range(visual_cols):
            # Cell border
            rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='gray', linewidth=1)
            ax.add_patch(rect)

            # Top left
            if i == 0 and j == 0:
                ax.text(j + 0.5, i + 0.5, "D", ha='center', va='center', fontweight='bold')

            # First row will contain seq2
            elif i == 0 and j >= 2:
                if j - 2 < len(seq2):
                    ax.text(j + 0.5, i + 0.5, f"{seq2[j - 2]}{j - 1}", ha='center', va='center')

            elif i >= 2 and j == 0:
                if i - 2 < len(seq1):
                    ax.text(j + 0.5, i + 0.5, f"{seq1[i - 2]}{i - 1}", ha='center', va='center')

            # Add score value in each cell - contains other score from score matrix
            elif i >= 1 and j >= 1:
                score_i = i - 1
                score_j = j - 1
                if score_i < rows and score_j < cols:
                    ax.text(j + 0.5, i + 0.5, f"{score_matrix[score_i, score_j]}",
                            ha='center', va='center', fontsize=11)

    # Create patches for the legend
    path_patches = []

    # Draw each optimal path with a unique color
    for path_idx, path in enumerate(all_paths):
        color = colors[path_idx]
        path_set = set((y + 1, x + 1) for y, x in path)

        for i, j in path_set:
            rect_fill = plt.Rectangle((j, i), 1, 1, fill=True, facecolor=color, alpha=0.3)
            ax.add_patch(rect_fill)

        # Add to legend
        path_patches.append(mpatches.Patch(color=color, alpha=0.3,
                                           label=f'Optimal Path {path_idx + 1}'))

    # Calculate alignment score (value at bottom-right cell)
    alignment_score = score_matrix[rows - 1, cols - 1]
    ax.text(1, rows + 1.5, f"Score: {alignment_score}", ha='center', va='center',
            fontsize=12, fontweight='bold', color='gray')

    # Add legend
    if path_patches:
        plt.legend(handles=path_patches, loc='upper center',
               bbox_to_anchor=(0.5, -0.05), ncol=min(5, len(all_paths)))

    # Set limits and remove ticks
    ax.set_xlim(-0.1, visual_cols + 0.1)
    ax.set_ylim(visual_rows + 0.1, -0.1)  # Inverted y-axis to match matrix orientation
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove borders
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.title(f"All Optimal Alignment Paths ({len(all_paths)} found)")
    plt.tight_layout()
    plt.show()


def convert_path_to_alignment(path, seq1, seq2):
    """
    Convert a path to an aligned sequence pair

    Args:
        path: List of coordinates representing the path
        seq1: The first sequence
        seq2: The second sequence

    Returns:
        tuple: The aligned sequences
    """
    path = list(reversed(path))
    aligned1 = []
    aligned2 = []

    for (i, j), (ni, nj) in zip(path, path[1:]):
        di, dj = ni - i, nj - j

        if di == 1 and dj == 1:
            # diagonal → consume one from each
            aligned1.append(seq1[i])  # i goes 0→len(seq1)-1
            aligned2.append(seq2[j])
        elif di == 1 and dj == 0:
            # down → gap in seq2
            aligned1.append(seq1[i])
            aligned2.append('-')
        elif di == 0 and dj == 1:
            # right → gap in seq1
            aligned1.append('-')
            aligned2.append(seq2[j])
        else:
            # should never happen in a correct backtrack
            raise ValueError(f"unexpected step: {(i, j)}→{(ni, nj)}")

    return ''.join(aligned1), ''.join(aligned2)


def run_needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-1, show_all_paths=False):
    """
    Run the complete Needleman-Wunsch algorithm workflow

    Args:
        seq1 (str): First sequence
        seq2 (str): Second sequence
        match (int): Match score
        mismatch (int): Mismatch score
        gap (int): Gap penalty
        show_all_paths (bool): Whether to find and display all optimal paths

    Returns:
        tuple: The aligned sequences and statistics
    """
    # Calculate score matrix and direction matrix
    score_matrix, direction_matrix = scoring_path(seq1, seq2, match, mismatch, gap)

    # Get the traceback path for one optimal alignment
    aligned_seq1, aligned_seq2, path_coords = traceback_alignment(seq1, seq2, direction_matrix)

    # Calculate alignment statistics
    parameters = {"match": match, "mismatch": mismatch, "gap": gap}
    statistics = calculate_matching_statistics(aligned_seq1, aligned_seq2)

    # Display the results
    display_alignment_result(aligned_seq1, aligned_seq2, statistics, parameters)

    # Visualize the alignment matrix with the optimal path
    visualize_alignment(seq1, seq2, score_matrix, path_coords)

    # Additional task: Find all optimal paths
    if show_all_paths:
        all_paths = find_all_optimal_paths(score_matrix, direction_matrix, seq1, seq2)
        print(f"\nFound {len(all_paths)} optimal alignment path(s).")

        # Visualize all optimal paths
        visualize_all_optimal_paths(seq1, seq2, score_matrix, all_paths)

        # Convert each path to an alignment and display
        if len(all_paths) > 1:
            print("\nAll Optimal Alignments:")
            for i, path in enumerate(all_paths):
                path_aligned_seq1, path_aligned_seq2 = convert_path_to_alignment(path, seq1, seq2)
                print(f"\nOptimal Alignment {i + 1}:")
                print(f"Seq1: {path_aligned_seq1}")
                print(
                    f"      {''.join(['|' if path_aligned_seq1[j] == path_aligned_seq2[j] and path_aligned_seq1[j] != '-' and path_aligned_seq2[j] != '-' else ' ' for j in range(len(path_aligned_seq1))])}")
                print(f"Seq2: {path_aligned_seq2}")

    return aligned_seq1, aligned_seq2, statistics, parameters, score_matrix, direction_matrix


def main():
    """
    Main function to run the sequence alignment program
    """
    parser = argparse.ArgumentParser(description='Needleman-Wunsch Sequence Alignment Tool')
    parser.add_argument('--fasta', type=str, help='Path to FASTA file with two sequences')
    parser.add_argument('--seq1', type=str, help='First sequence (if not using FASTA)')
    parser.add_argument('--seq2', type=str, help='Second sequence (if not using FASTA)')
    parser.add_argument('--match', type=int, default=1, help='Score for matches (default: 1)')
    parser.add_argument('--mismatch', type=int, default=-1, help='Score for mismatches (default: -1)')
    parser.add_argument('--gap', type=int, default=-1, help='Penalty for gaps (default: -1)')
    parser.add_argument('--output', type=str, help='Output file for alignment results')
    parser.add_argument('--dotplot', action='store_true', help='Generate a dotplot visualization')
    parser.add_argument('--window', type=int, default=1, help='Window size for dotplot (default: 1)')
    parser.add_argument('--threshold', type=int, default=1, help='Threshold for dotplot (default: 1)')
    parser.add_argument('--all_paths', action='store_true', help='Find and display all optimal paths')

    args = parser.parse_args()

    match, mismatch, gap = 1, -1, -1

    #
    # Interactive menu if no arguments provided
    #
    if len(sys.argv) == 1:
        print("\n" + "=" * 60)
        print("NEEDLEMAN-WUNSCH SEQUENCE ALIGNMENT TOOL")
        print("=" * 60)

        while True:
            print("\nMenu Options:")
            print("1. Load sequences manually")
            print("2. Load sequences from FASTA file")
            print("3. Exit")

            choice = input("\nEnter your choice (1-3): ")

            if choice == '1':
                # print("\nEnter the sequences:")
                # seq_type = input("Sequence type (DNA/PROTEIN) [DNA]: ").strip().upper() or "DNA"
                # if seq_type not in ["DNA", "PROTEIN"]:
                #     print("Invalid sequence type. Using DNA as default.")
                #     seq_type = "DNA"

                seq1 = input("Enter sequence 1: ").strip().upper()
                seq2 = input("Enter sequence 2: ").strip().upper()

                if not seq1 or not seq2:
                    print("Error: Sequences cannot be empty.")
                    continue

                # if not validate_sequence(seq1, seq_type) or not validate_sequence(seq2, seq_type):
                #     print(f"Error: Invalid {seq_type} sequence(s).")
                #     continue

            elif choice == '2':
                fasta_path = input("Enter path to FASTA file: ").strip()
                try:
                    seq1, seq2 = load_sequence_fasta(fasta_path)
                    # seq_type = input("Sequence type (DNA/PROTEIN) [DNA]: ").strip().upper() or "DNA"
                    # if seq_type not in ["DNA", "PROTEIN"]:
                    #     print("Invalid sequence type. Using DNA as default.")
                    #     seq_type = "DNA"

                    # if not validate_sequence(seq1, seq_type) or not validate_sequence(seq2, seq_type):
                    #     print(f"Error: Invalid {seq_type} sequence(s) in the FASTA file.")
                    #     continue

                except Exception as e:
                    print(f"Error loading FASTA file: {e}")
                    continue
            elif choice == '3':
                print("Exiting program. Goodbye!")
                sys.exit(0)
            else:
                print("Invalid choice. Please select 1-3.")
                continue

            # Get scoring parameters
            print("\nEnter scoring parameters:")
            try:
                match = int(input("Match score [1]: ") or "1")
                mismatch = int(input("Mismatch score [-1]: ") or "-1")
                gap = int(input("Gap penalty [-1]: ") or "-1")
            except ValueError:
                print("Invalid input. Using default scoring parameters.")
                match, mismatch, gap = 1, -1, -1

            # Ask for dotplot
            generate_dotplot = input("\nGenerate dotplot? (y/n) [n]: ").strip().lower() == 'y'

            if generate_dotplot:
                try:
                    window = int(input("Window size [1]: ") or "1")
                    threshold = int(input("Threshold [1]: ") or "1")
                except ValueError:
                    print("Invalid input. Using default dotplot parameters.")
                    window, threshold = 1, 1

                # Generate dotplot
                dp = dotplot(seq1, seq2, window, threshold)
                dotplotGraphic(dp, seq1, seq2, "Sequence Similarity Dotplot")

            # Run Needleman-Wunsch algorithm
            show_all_paths = input("\nFind all optimal paths? (y/n) [n]: ").strip().lower() == 'y'

            aligned_seq1, aligned_seq2, statistics, parameters, score_matrix, direction_matrix = run_needleman_wunsch(
                seq1, seq2, match, mismatch, gap, show_all_paths)

            # Save results to file
            save_results = input("\nSave alignment results to file? (y/n) [n]: ").strip().lower() == 'y'
            if save_results:
                output_file = input("Enter output filename [alignment_result.txt]: ").strip() or "alignment_result.txt"

                extra = None
                if show_all_paths:
                    all_paths = find_all_optimal_paths(score_matrix, direction_matrix, seq1, seq2)
                    extra = []
                    for idx, path in enumerate(all_paths, 1):
                        a1, a2 = convert_path_to_alignment(path, seq1, seq2)
                        extra.append((a1, a2))

                save_alignment_to_file(aligned_seq1, aligned_seq2, statistics, parameters, output_file, extra)

            # Back to menu or exit
            continue_choice = input("\nReturn to main menu? (y/n) [y]: ").strip().lower() or 'y'
            if continue_choice != 'y':
                print("Exiting program. Goodbye!")
                break

    else:
        # Command-line mode
        if args.fasta:
            try:
                seq1, seq2 = load_sequence_fasta(args.fasta)
            except Exception as e:
                print(f"Error loading FASTA file: {e}")
                sys.exit(1)
        elif args.seq1 and args.seq2:
            seq1 = args.seq1.upper()
            seq2 = args.seq2.upper()
        else:
            print("Error: Either provide a FASTA file or both sequences.")
            parser.print_help()
            sys.exit(1)

        # # Validate sequences
        # if not validate_sequence(seq1, args.type) or not validate_sequence(seq2, args.type):
        #     print(f"Error: Invalid {args.type} sequence(s).")
        #     sys.exit(1)

        # Generate dotplot if requested
        if args.dotplot:
            dp = dotplot(seq1, seq2, args.window, args.threshold)
            dotplotGraphic(dp, seq1, seq2, "Sequence Similarity Dotplot")

        # Run Needleman-Wunsch algorithm
        aligned_seq1, aligned_seq2, statistics, parameters, score, direction = run_needleman_wunsch(
            seq1, seq2, args.match, args.mismatch, args.gap, args.all_paths)

        score_matrix, direction_matrix = scoring_path(seq1, seq2, match, mismatch, gap)

        # Save results to file if requested
        if args.output:

            extra = None
            if args.all_paths:
                all_paths = find_all_optimal_paths(score_matrix, direction_matrix, seq1, seq2)
                extra = [convert_path_to_alignment(path, seq1, seq2) for path in all_paths]
                # for idx, path in enumerate(all_paths, 1):
                #     a1, a2 = convert_path_to_alignment(path, seq1, seq2)
                #     extra.append((f"Optimal alignment {idx}", a1, a2))

            save_alignment_to_file(aligned_seq1, aligned_seq2, statistics, parameters, args.output, extra)


if __name__ == "__main__":
    main()