"""Sharp finite-archive comparisons for two post-answer cascade policies."""
import itertools
import math
from statistics import fmean


def assignment(matrix):
    """Minimum-weight square assignment, using shortest augmenting paths."""
    size = len(matrix)
    if not size or any(len(row) != size for row in matrix):
        raise ValueError("assignment needs a nonempty square matrix")
    left, right = [0.] * (size+1), [0.] * (size+1)
    match, parent = [0] * (size+1), [0] * (size+1)
    for row in range(1, size+1):
        match[0] = row
        col = 0
        slack = [math.inf] * (size+1)
        visited = [False] * (size+1)
        while True:
            visited[col] = True
            i = match[col]
            for j in range(1, size+1):
                if not visited[j]:
                    reduced = matrix[i-1][j-1] - left[i] - right[j]
                    if reduced < slack[j]:
                        slack[j], parent[j] = reduced, col
            nxt = min((j for j in range(1,size+1) if not visited[j]), key=slack.__getitem__)
            step = slack[nxt]
            for j in range(size+1):
                if visited[j]:
                    left[match[j]] += step
                    right[j] -= step
                else:
                    slack[j] -= step
            col = nxt
            if match[col] == 0:
                break
        while col:
            prev = parent[col]
            match[col] = match[prev]
            col = prev
    perm = [0] * size
    for j in range(1, size+1):
        perm[match[j]-1] = j-1
    return perm


def erase_count(identity, losses):
    """Least number of individually rematched questions reaching the opposite weak sign."""
    if abs(identity) < 1e-10:
        return 0
    remaining = abs(identity)
    for count, loss in enumerate(sorted(losses, reverse=True), 1):
        remaining -= loss
        if remaining <= 1e-10:
            return count
    return None


def compare(cells):
    """Cells contain (gate-probability differences, fallback utilities, constant sum).

    Each cell is equally weighted, each index uniform. Utilities and constants use
    the same units. The constant is summed over first-stage replicates.
    """
    n = len(cells)
    if not n:
        raise ValueError("empty comparison")
    m = len(cells[0][0])
    if m < 2:
        raise ValueError("at least two replicate positions required")
    matrix = [[0.] * m for _ in range(m)]
    fixed = 0.
    summaries = []
    variance_sum = 0.
    for weights, values, constant in cells:
        if len(weights) != m or len(values) != m:
            raise ValueError("common reindexing requires aligned replicate slots")
        fixed += constant
        w, v = sorted(weights), sorted(values)
        lower = sum(a*b for a,b in zip(w,reversed(v))) + constant
        upper = sum(a*b for a,b in zip(w,v)) + constant
        identity = sum(a*b for a,b in zip(weights,values)) + constant
        summaries.append((lower, identity, upper))
        mw, mv = fmean(w), fmean(v)
        variance_sum += sum((a-mw)**2 for a in w)*sum((b-mv)**2 for b in v)/(m-1)
        for i, a in enumerate(weights):
            if a:
                row = matrix[i]
                for j,b in enumerate(values):
                    row[j] += a*b
    scale = n*m
    def score(perm):
        return (fixed+sum(row[j] for row,j in zip(matrix,perm)))/scale
    lo = assignment(matrix)
    hi = assignment([[-v for v in row] for row in matrix])
    grand = fmean(map(fmean,matrix))
    row_mean = list(map(fmean,matrix))
    col_mean = list(map(fmean,zip(*matrix)))
    centered_ss = sum((x-row_mean[i]-col_mean[j]+grand)**2
                      for i,row in enumerate(matrix) for j,x in enumerate(row))
    low_sum, identity_sum, high_sum = map(sum,zip(*summaries))
    losses = [identity-low if identity_sum>0 else high-identity for low,identity,high in summaries]
    signs = None
    if m <= 4:
        scores = [score(p) for p in itertools.permutations(range(m))]
        signs = {"negative":sum(x < -1e-10 for x in scores), "zero":sum(abs(x)<=1e-10 for x in scores),
                 "positive":sum(x>1e-10 for x in scores)}
    return dict(global_permutation_signs=signs, local_min=low_sum/scale, local_max=high_sum/scale,
                common_min=score(lo), common_max=score(hi), identity=identity_sum/scale,
                product=(fixed+m*grand)/scale, common_sd=math.sqrt(centered_ss/(m-1))/scale,
                local_sd=math.sqrt(variance_sum)/scale, erase_questions=erase_count(identity_sum,losses),
                common_min_permutation=lo, common_max_permutation=hi)
