# Needleman-Wunsch Sequence Alignment Tool

## Introduction

This program implements the Needleman-Wunsch algorithm for global alignment of protein or DNA sequences. It provides both a command-line interface and an interactive menu-driven interface for ease of use.

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- Seaborn

You can install the required packages using pip:

```bash
pip install numpy matplotlib
```

## Files

- `main.py`: The main application with interactive menu and command-line interface
- `task2.py`: Contains the core algorithm implementations

## Features

1. **Sequence Input**:
   - Manual entry of sequences
   - Loading sequences from FASTA files

2. **Alignment Parameters**:
   - Customizable match score
   - Customizable mismatch penalty
   - Customizable gap penalty

3. **Visualization**:
   - Dot plot visualization of sequence similarity
   - Graphical display of the scoring matrix with optimal alignment path
   - Visualization of all optimal alignment paths (additional task)

4. **Output**:
   - Display of alignment results with statistics
   - Saving results to text files

## Usage Instructions

### Interactive Menu

Run the program without any arguments to access the interactive menu:

```bash
python main.py
```

This will present you with options to:
1. Load sequences manually
2. Load sequences from a FASTA file
3. Exit

After loading sequences, you can:
- Set scoring parameters
- Generate a dot plot visualization
- Find all optimal alignment paths
- Save results to a file

### Command-Line Interface

You can also run the program with command-line arguments:

```bash
python main.py --seq1 ACGCACTA --seq2 ACTGATTA --match 1 --mismatch -1 --gap -1 --output results.txt --dotplot --all_paths
```

Available options:
- `--fasta`: Path to a FASTA file containing two sequences
- `--seq1`, `--seq2`: Input sequences directly
- `--match`: Score for matches (default: 1)
- `--mismatch`: Score for mismatches (default: -1)
- `--gap`: Penalty for gaps (default: -1)
- `--output`: Output file for alignment results
- `--dotplot`: Generate a dot plot visualization
- `--window`: Window size for dot plot (default: 1)
- `--threshold`: Threshold for dot plot (default: 1)
- `--all_paths`: Find and display all optimal paths
- `--type`: Sequence type ('DNA' or 'PROTEIN', default: 'DNA')

## Example

For a quick test, run the program with the provided example sequences:

```bash
python main.py --seq1 ACGCACTA --seq2 ACTGATTA --output results.txt --dotplot --all_paths
```

## Computational Complexity

- **Time Complexity**: O(m*n), where m and n are the lengths of the two sequences
- **Space Complexity**: O(m*n) for storing the scoring matrix and direction matrix

## Additional Task: Finding All Optimal Paths

The program can find and visualize all optimal alignment paths with the same maximum score. This is activated using the `--all_paths` option in command-line mode or by selecting the corresponding option in the interactive menu.