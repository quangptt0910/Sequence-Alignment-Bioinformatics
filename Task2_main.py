import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

# =====================
# 1. Sequence I/O
# =====================

def load_sequence_manual():
    """
    Prompt the user to enter two sequences interactively.
    Returns: seq1 (str), seq2 (str)
    """
    seq1 = input("Enter sequence 1: ").strip().upper()
    seq2 = input("Enter sequence 2: ").strip().upper()
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

# =====================
# 2. Scoring Parameters
# =====================

def get_scoring_params():
    """
    Prompt user for match, mismatch, and gap penalties.
    Returns:
        match_score (int), mismatch_score (int), gap_penalty (int)
    """
    print("-- Scoring parameters --")
    m = int(input("Match score (e.g. +1): "))
    mm = int(input("Mismatch score (e.g. 0): "))
    g = int(input("Gap penalty (e.g. -1): "))
    return m, mm, g

# =====================
# 3. Alignment (Needleman-Wunsch)
# =====================

def compute_alignment(seq1, seq2, match, mismatch, gap):
    """
    Compute scoring and traceback matrices.
    Returns:
        score_matrix, direction_matrix
    """
    len1, len2 = len(seq1), len(seq2)
    score = np.zeros((len1+1, len2+1), dtype=int)
    direction = np.zeros_like(score)
    # initialize
    for i in range(1, len1+1):
        score[i,0] = score[i-1,0] + gap
        direction[i,0] = 1  # up
    for j in range(1, len2+1):
        score[0,j] = score[0,j-1] + gap
        direction[0,j] = 2  # left
    # fill
    for i in range(1, len1+1):
        for j in range(1, len2+1):
            diag = score[i-1,j-1] + (match if seq1[i-1]==seq2[j-1] else mismatch)
            up   = score[i-1,j] + gap
            left = score[i,j-1] + gap
            best = max(diag, up, left)
            score[i,j] = best
            if best == diag:
                direction[i,j] = 0
            elif best == up:
                direction[i,j] = 1
            else:
                direction[i,j] = 2
    return score, direction

# =====================
# 4. Traceback & Stats
# =====================

def traceback_one(seq1, seq2, direction):
    """
    Recover one optimal alignment from direction matrix.
    Returns:
        aligned1, aligned2, path_coords
    """
    i, j = len(seq1), len(seq2)
    a1, a2 = [], []
    path = [(i,j)]
    while i>0 or j>0:
        d = direction[i,j]
        if i>0 and j>0 and d==0:
            a1.append(seq1[i-1]); a2.append(seq2[j-1])
            i-=1; j-=1
        elif i>0 and d==1:
            a1.append(seq1[i-1]); a2.append('-'); i-=1
        else:
            a1.append('-'); a2.append(seq2[j-1]); j-=1
        path.append((i,j))
    return ''.join(reversed(a1)), ''.join(reversed(a2)), list(reversed(path))


def compute_stats(al1, al2):
    length = len(al1)
    matches = sum(a==b for a,b in zip(al1,al2))
    gaps = sum(a=='-' or b=='-' for a,b in zip(al1,al2))
    pid = matches/length*100
    pg  = gaps/length*100
    return length, pid, pg

# =====================
# 5. Output & Visualization
# =====================

def save_results(filename, seq1, seq2, params, score_matrix, alignments, stats):
    with open(filename,'w') as f:
        f.write(f"Params: match={params[0]}, mismatch={params[1]}, gap={params[2]}\n")
        f.write(f"Seq1: {seq1}\nSeq2: {seq2}\n")
        for idx,(al1,al2) in enumerate(alignments,1):
            length,pid,pg = stats[idx-1]
            f.write(f"\nPath {idx}: length={length}, identity={pid:.2f}%, gaps={pg:.2f}%\n")
            f.write(''.join(al1) + "\n" + ''.join(al2) + "\n")


def plot_scoreboard(score_matrix, path, seq1, seq2):
    fig, ax = plt.subplots(figsize=(len(seq2)+1, len(seq1)+1))
    ax.imshow(score_matrix, cmap='Blues', origin='upper')
    ys, xs = zip(*path)
    ax.plot(xs, ys, marker='o', color='red')
    ax.set_xticks(range(len(seq2)+1)); ax.set_xticklabels(['-']+list(seq2))
    ax.set_yticks(range(len(seq1)+1)); ax.set_yticklabels(['-']+list(seq1))
    plt.show()

# =====================
# 6. Additional: All Optimal Paths
# =====================

def traceback_all(seq1, seq2, direction):
    paths=[]
    def backtrack(i,j,a1,a2):
        if i==0 and j==0:
            paths.append((a1[::-1],a2[::-1])); return
        d=direction[i,j]
        if i>0 and j>0 and d==0:
            backtrack(i-1,j-1,a1+[seq1[i-1]],a2+[seq2[j-1]])
        if i>0 and d==1:
            backtrack(i-1,j,a1+[seq1[i-1]],a2+['-'])
        if j>0 and d==2:
            backtrack(i,j-1,a1+['-'],a2+[seq2[j-1]])
    backtrack(len(seq1),len(seq2),[],[])
    return paths

# =====================
# Main Application
# =====================

def main():
    parser = argparse.ArgumentParser(
        description="Quantitative matching of sequence pairs (Needleman-Wunsch)"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-m','--manual',action='store_true',help='Enter sequences manually')
    input_group.add_argument('--fasta1',type=str,help='FASTA file for sequence 1')
    parser.add_argument('--fasta2',type=str,help='FASTA file for sequence 2 (use with --fasta1)')
    parser.add_argument('-a','--all-paths',action='store_true',help='Generate all optimal paths')
    parser.add_argument('-o','--output',type=str,default='results.txt',help='Output text file')
    args=parser.parse_args()

    if args.manual:
        seq1, seq2 = load_sequence_manual()
    else:
        if not args.fasta1 or not args.fasta2:
            parser.error("--fasta1 and --fasta2 must both be provided when loading from FASTA")
        seq1, seq2 = load_sequence_fasta(args.fasta1, args.fasta2)

    match, mismatch, gap = get_scoring_params()
    score_matrix, direction = compute_alignment(seq1,seq2,match,mismatch,gap)

    alignments=[]; stats=[]
    if args.all_paths:
        for al1,al2 in traceback_all(seq1,seq2,direction):
            alignments.append((al1,al2))
            stats.append(compute_stats(al1,al2))
    else:
        al1,al2,path = traceback_one(seq1,seq2,direction)
        alignments.append((al1,al2)); stats.append(compute_stats(al1,al2))

    for idx,(al1,al2) in enumerate(alignments,1):
        length,pid,pg = stats[idx-1]
        print(f"Path {idx}: length={length}, identity={pid:.2f}%, gaps={pg:.2f}%")
    save_results(args.output, seq1, seq2, (match,mismatch,gap), score_matrix, alignments, stats)

    # Plot only first path
    if not args.all_paths:
        plot_scoreboard(score_matrix, path, seq1, seq2)

if __name__=='__main__':
    main()
