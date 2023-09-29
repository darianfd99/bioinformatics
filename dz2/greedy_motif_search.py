from typing import List
import numpy as np

nucleotides = 'ACTG'
nucleotides_dict = {}
for i, nucleotide in enumerate(nucleotides):
    nucleotides_dict[nucleotide] = i


def get_profile(motifs: List[str]) -> np.array:
    k = len(motifs[0])
    t = len(motifs)
    profile = np.zeros((4, k), dtype=np.float64)
    for motif in motifs:
        for i, nucleotide in enumerate(motif):
            profile[nucleotides_dict[nucleotide], i] += 1 / t

    return profile


def get_most_probable_motif(dna: str, profile: np.array) -> str:
    k = profile.shape[1]
    best_motif = dna[0:k]
    best_prob = -1

    for i in range(len(dna) - k + 1):
        prob = 1
        motif = dna[i: i + k]
        for j, nucleotide in enumerate(motif):
            prob *= profile[nucleotides_dict[nucleotide], j]
        if prob > best_prob:
            best_prob = prob
            best_motif = motif

    return best_motif


def get_score(motifs: List[str]) -> float:
    profile = get_profile(motifs)
    return np.sum(np.sum(profile, axis=0) - np.max(profile, axis=0))


def greedy_motif_search(dna: List[str], k: int, t: int) -> List[str]:
    best_motifs = [dna[i][0:k] for i in range(len(dna))]
    best_score = get_score(best_motifs)

    for i in range(len(dna[0]) - k + 1):
        motifs = [dna[0][i:i + k]]
        for j in range(1, t):
            profile = get_profile(motifs)
            motif = get_most_probable_motif(dna[j], profile)
            motifs.append(motif)

        new_score = get_score(motifs)
        if new_score < best_score:
            best_score = new_score
            best_motifs = motifs.copy()

    return best_motifs


if __name__ == '__main__':
    dna = []
    with open('rosalind.txt') as f:
        line = f.readline()
        args = line.split(' ', 3)
        args = args[:2]
        k, t = int(args[0]), int(args[1])
        for i in range(t):
            line = f.readline()[:-1]
            dna.append(line)

    li = greedy_motif_search(dna, k, t)
    print('\n'.join(li))
