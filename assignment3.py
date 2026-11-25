# DAFERMOU CHRISTINA DANAE - 5206
# VALERAS CHRISTOS - 5182

import sys
import math
import time
import heapq
import bisect

# ===========================================================================================
# Program Description and Usage
# This program implements three methods for similarity search over a dataset
# of 10000 objects in 10 dimensions, using the Euclidean (L2) distance.
#   1) Naive linear scan
#   2) Pivot-based search (triangle inequality pruning)
#   3) iDistance search (sorted 1D index + binary search)
# Both range queries and kNN queries are supported.

# The program is executed from the command line with optional arguments:
#   python assignment3.py <numpivots> <eps> <k> [data_file] [query_file]
# ===========================================================================================


# ===========================================================================================
# Global Distance Counter
# A single counter is used to measure how many distance computations each method performs.
# Every method calls dist_counted() so the counter is increased automatically.
# ===========================================================================================

dist_count = 0


def dist_counted(a, b):
    # Euclidean (L2) distance between two vectors, counted in the global counter.
    global dist_count
    dist_count += 1
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s = s + d * d
    return math.sqrt(s)


def dist(a, b):
    # Euclidean (L2) distance, NOT counted (used during pivot/index construction).
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s = s + d * d
    return math.sqrt(s)


# ===========================================================================================
# MEROS 1: Data Loading and Pivot Computation
# The code reads the data file into a list of vectors D, where D[i] is the object with id i.
# Then it selects numpivots pivots using the farthest-point procedure:
#   - seed       = the first object (id 0)
#   - 1st pivot  = the object farthest from the seed
#   - k-th pivot = the object that maximizes the sum of distances to all previous pivots
# It also builds a 2D array distances[i][j] = distance from object i to pivot j.
# ===========================================================================================


def load_data(filename):
    D = []
    f = open(filename, "r")
    for line in f:
        line = line.strip()
        if line == "":
            continue
        # split on whitespace and convert every token to float
        parts = line.split()
        vec = []
        for x in parts:
            # keep only valid float characters (protects against stray junk like a BOM)
            clean = "".join(ch for ch in x if ch.isdigit() or ch in ".-+eE")
            if clean == "":
                continue
            vec.append(float(clean))
        D.append(vec)
    f.close()
    return D


def compute_pivots(D, numpivots):
    n = len(D)
    pivots = []

    # ---- 1st pivot: the object farthest from the seed (D[0]) ----
    seed = D[0]
    best_id = -1
    best_d = -1.0
    for i in range(n):
        d = dist(seed, D[i])
        if d > best_d:
            best_d = d
            best_id = i
    pivots.append(best_id)

    # ---- remaining pivots: maximize the sum of distances to all chosen pivots ----
    # sum_dists[i] keeps the running sum of distances of object i to all current pivots

    sum_dists = [0.0] * n
    for i in range(n):
        sum_dists[i] = dist(D[i], D[pivots[0]])

    pivot_set = set(pivots)  # fast membership test
    while len(pivots) < numpivots:
        best_id = -1
        best_d = -1.0
        for i in range(n):
            if i in pivot_set:
                continue
            if sum_dists[i] > best_d:
                best_d = sum_dists[i]
                best_id = i
        pivots.append(best_id)
        pivot_set.add(best_id)
        # update the running sums with the distance to the new pivot
        for i in range(n):
            sum_dists[i] = sum_dists[i] + dist(D[i], D[best_id])

    # ---- 2D array of distances: distances[i][j] = dist(object i, pivot j) ----
    distances = []
    for i in range(n):
        row = []
        for j in range(numpivots):
            row.append(dist(D[i], D[pivots[j]]))
        distances.append(row)

    return pivots, distances


# ===========================================================================================
# MEROS 2: iDistance Index Construction
# For every object the code finds its nearest pivot and records that distance.
# For every pivot p it keeps maxd(p) = the largest distance among the objects
# whose nearest pivot is p. The global maxd = max over all maxd(p).
# Each object gets an iDistance value:  iDist(o) = i * maxd + dist(o, p_i),
# where p_i is the nearest pivot of o. The (iDist, oid) pairs are sorted by iDist.
# ===========================================================================================


def build_idistance(distances, numpivots):
    n = len(distances)
    nearest_pivot = [0] * n
    nearest_dist = [0.0] * n
    maxd_p = [0.0] * numpivots  # max distance per pivot (over its own objects)

    for i in range(n):
        # find the nearest pivot of object i
        bp = 0
        bd = distances[i][0]
        for j in range(1, numpivots):
            if distances[i][j] < bd:
                bd = distances[i][j]
                bp = j
        nearest_pivot[i] = bp
        nearest_dist[i] = bd
        if bd > maxd_p[bp]:
            maxd_p[bp] = bd

    # global maximum over all maxd(p)
    maxd = 0.0
    for j in range(numpivots):
        if maxd_p[j] > maxd:
            maxd = maxd_p[j]

    # build the (iDist, oid) array
    idist_array = []
    for i in range(n):
        i_val = nearest_pivot[i] * maxd + nearest_dist[i]
        idist_array.append((i_val, i))

    # sort by iDist value
    idist_array.sort(key=lambda x: x[0])

    return idist_array, maxd_p, maxd, nearest_pivot


# ===========================================================================================
# MEROS 3: Range Queries
# A range query returns every object o with dist(q, o) <= eps.
# Three methods are provided: naive, pivot-based, and iDistance-based.
# ===========================================================================================


def range_naive(D, q, eps):
    result = []
    for i in range(len(D)):
        d = dist_counted(D[i], q)
        if d <= eps:
            result.append(i)
    return result


def range_pivot(D, q, eps, pivots, distances):
    # (b) prune with the triangle inequality before computing the real distance
    numpivots = len(pivots)
    # distance of q to each pivot, computed once
    qp = []
    for j in range(numpivots):
        qp.append(dist_counted(D[pivots[j]], q))

    result = []
    for i in range(len(D)):
        prune = False
        for j in range(numpivots):
            # if |dist(pi,o) - dist(pi,q)| > eps  then o cannot be within eps
            if abs(distances[i][j] - qp[j]) > eps:
                prune = True
                break
        if not prune:
            d = dist_counted(D[i], q)
            if d <= eps:
                result.append(i)
    return result


def range_idistance(D, q, eps, pivots, idist_array, maxd, nearest_pivot, keys):
    # (c) use the sorted iDistance array and binary search per partition
    numpivots = len(pivots)
    # distance of q to each pivot, computed once
    qp = []
    for j in range(numpivots):
        qp.append(dist_counted(D[pivots[j]], q))

    result = []
    for j in range(numpivots):
        # candidate iDistance window for the partition of pivot j
        lo = j * maxd + qp[j] - eps
        hi = j * maxd + qp[j] + eps

        # binary search for the first element >= lo, then scan until hi
        start = bisect.bisect_left(keys, lo)
        k = start
        while k < len(idist_array) and keys[k] <= hi:
            oid = idist_array[k][1]
            # only objects whose nearest pivot is j belong to this partition
            if nearest_pivot[oid] == j:
                d = dist_counted(D[oid], q)
                if d <= eps:
                    result.append(oid)
            k += 1

    return result


# ===========================================================================================
# MEROS 4: kNN Queries
# A kNN query returns the k objects with the smallest dist(q, o).
# A max-heap (implemented with negative keys on heapq) keeps the current k best objects.
# The top of the heap is the current k-th distance and acts as a dynamic eps for pruning.
# ===========================================================================================


def knn_naive(D, q, k):
    # (a) scan everything, keep the k closest with a max-heap
    heap = []  # elements: (-dist, oid)
    for i in range(len(D)):
        d = dist_counted(D[i], q)
        if len(heap) < k:
            heapq.heappush(heap, (-d, i))
        else:
            if d < -heap[0][0]:
                heapq.heapreplace(heap, (-d, i))
    # return the k nearest sorted from closest to farthest
    res = []
    while heap:
        nd, oid = heapq.heappop(heap)
        res.append((-nd, oid))
    res.reverse()
    return res


def knn_pivot(D, q, k, pivots, distances):
    # (b) fill the heap first, then use the k-th distance as eps for pivot pruning
    numpivots = len(pivots)
    qp = []
    for j in range(numpivots):
        qp.append(dist_counted(D[pivots[j]], q))

    heap = []  # max-heap with negative keys
    for i in range(len(D)):
        if len(heap) < k:
            d = dist_counted(D[i], q)
            heapq.heappush(heap, (-d, i))
        else:
            eps = -heap[0][0]  # current k-th distance
            prune = False
            for j in range(numpivots):
                if abs(distances[i][j] - qp[j]) > eps:
                    prune = True
                    break
            if not prune:
                d = dist_counted(D[i], q)
                if d < eps:
                    heapq.heapreplace(heap, (-d, i))

    res = []
    while heap:
        nd, oid = heapq.heappop(heap)
        res.append((-nd, oid))
    res.reverse()
    return res


def knn_idistance(D, q, k, pivots, idist_array, maxd, nearest_pivot, partition_objs, keys):
    # (c) start from the partition of the nearest pivot, then try to prune the rest
    numpivots = len(pivots)
    qp = []
    for j in range(numpivots):
        qp.append(dist_counted(D[pivots[j]], q))

    # nearest pivot of the query point
    q_npiv = 0
    for j in range(1, numpivots):
        if qp[j] < qp[q_npiv]:
            q_npiv = j

    # ---- Step 1: naive scan over the partition of the nearest pivot ----
    heap = []
    for oid in partition_objs[q_npiv]:
        d = dist_counted(D[oid], q)
        if len(heap) < k:
            heapq.heappush(heap, (-d, oid))
        else:
            if d < -heap[0][0]:
                heapq.heapreplace(heap, (-d, oid))

    # ---- Step 2: for the remaining pivots, prune their partitions if possible ----
    for j in range(numpivots):
        if j == q_npiv:
            continue

        # if the heap is not full yet, we cannot prune (eps is infinite)
        if len(heap) < k:
            for oid in partition_objs[j]:
                d = dist_counted(D[oid], q)
                if len(heap) < k:
                    heapq.heappush(heap, (-d, oid))
                else:
                    if d < -heap[0][0]:
                        heapq.heapreplace(heap, (-d, oid))
            continue

        eps = -heap[0][0]  # current k-th distance, used as eps
        lo = j * maxd + qp[j] - eps
        hi = j * maxd + qp[j] + eps

        start = bisect.bisect_left(keys, lo)
        kk = start
        while kk < len(idist_array) and keys[kk] <= hi:
            oid = idist_array[kk][1]
            if nearest_pivot[oid] == j:
                d = dist_counted(D[oid], q)
                if len(heap) < k:
                    heapq.heappush(heap, (-d, oid))
                else:
                    if d < -heap[0][0]:
                        heapq.heapreplace(heap, (-d, oid))
            kk += 1

    res = []
    while heap:
        nd, oid = heapq.heappop(heap)
        res.append((-nd, oid))
    res.reverse()
    return res


# ===========================================================================================
# Main
# Runs all three methods for both range and kNN queries over the query file,
# measuring the total time and the average number of distance computations per query.
# ===========================================================================================


def main():
    global dist_count

    # ---- default values ----
    numpivots = 5
    eps = 0.2
    k = 5
    data_file = "data10K10.txt"
    query_file = "queries10.txt"

    # ---- optional command line arguments ----
    # python assignment3.py <numpivots> <eps> <k> [data_file] [query_file]
    if len(sys.argv) >= 2:
        numpivots = int(sys.argv[1])
    if len(sys.argv) >= 3:
        eps = float(sys.argv[2])
    if len(sys.argv) >= 4:
        k = int(sys.argv[3])
    if len(sys.argv) >= 5:
        data_file = sys.argv[4]
    if len(sys.argv) >= 6:
        query_file = sys.argv[5]

    # ---- MEROS 1: load data and compute pivots ----
    D = load_data(data_file)
    Q = load_data(query_file)
    t0 = time.time()
    pivots, distances = compute_pivots(D, numpivots)
    pivot_time = time.time() - t0

    # ---- MEROS 2: build the iDistance index ----
    idist_array, maxd_p, maxd, nearest_pivot = build_idistance(distances, numpivots)
    # precompute the iDist keys once (used by binary search in every query)
    keys = [x[0] for x in idist_array]
    # group object ids per partition (used by the iDistance kNN)
    partition_objs = [[] for _ in range(numpivots)]
    for i in range(len(D)):
        partition_objs[nearest_pivot[i]].append(i)

    print("Running queries... please wait")
    sys.stdout.flush()

    # ---- MEROS 3: range queries (three methods) ----
    dist_count = 0
    t0 = time.time()
    for q in Q:
        range_naive(D, q, eps)
    time_naive_range = time.time() - t0
    avg_naive_range = float(dist_count) / len(Q)

    dist_count = 0
    t0 = time.time()
    for q in Q:
        range_pivot(D, q, eps, pivots, distances)
    time_pivot_range = time.time() - t0
    avg_pivot_range = float(dist_count) / len(Q)

    dist_count = 0
    t0 = time.time()
    for q in Q:
        range_idistance(D, q, eps, pivots, idist_array, maxd, nearest_pivot, keys)
    time_idist_range = time.time() - t0
    avg_idist_range = float(dist_count) / len(Q)

    # ---- MEROS 4: kNN queries (three methods) ----
    dist_count = 0
    t0 = time.time()
    for q in Q:
        knn_naive(D, q, k)
    time_naive_knn = time.time() - t0
    avg_naive_knn = float(dist_count) / len(Q)

    dist_count = 0
    t0 = time.time()
    for q in Q:
        knn_pivot(D, q, k, pivots, distances)
    time_pivot_knn = time.time() - t0
    avg_pivot_knn = float(dist_count) / len(Q)

    dist_count = 0
    t0 = time.time()
    for q in Q:
        knn_idistance(D, q, k, pivots, idist_array, maxd, nearest_pivot, partition_objs, keys)
    time_idist_knn = time.time() - t0
    avg_idist_knn = float(dist_count) / len(Q)

    # ===========================================================================================
    # Output
    # ===========================================================================================
    n = "=" * 52

    print(n)
    print("MEROS 1: PIVOTS")
    print(n)
    print("  data file        : {0}".format(data_file))
    print("  objects loaded   : {0}".format(len(D)))
    print("  queries loaded   : {0}".format(len(Q)))
    print("  numpivots        : {0}".format(numpivots))
    print("  pivots           : {0}".format(pivots))
    print("  computation time : {0:.4f} s".format(pivot_time))

    print("")
    print(n)
    print("MEROS 2: iDISTANCE INDEX")
    print(n)
    print("  maxd (global)    : {0:.4f}".format(maxd))
    for j in range(numpivots):
        print("    pivot {0} (id={1}) -> maxd = {2:.4f}".format(j, pivots[j], maxd_p[j]))

    print("")
    print(n)
    print("MEROS 3: RANGE QUERIES (eps = {0})".format(eps))
    print(n)
    print("  {0:<14}{1:>16}{2:>16}".format("method", "avg dist comp", "time (s)"))
    print("  " + "-" * 46)
    print("  {0:<14}{1:>16.2f}{2:>16.4f}".format("Naive",       avg_naive_range, time_naive_range))
    print("  {0:<14}{1:>16.2f}{2:>16.4f}".format("Pivot-based", avg_pivot_range, time_pivot_range))
    print("  {0:<14}{1:>16.2f}{2:>16.4f}".format("iDistance",   avg_idist_range, time_idist_range))

    print("")
    print(n)
    print("MEROS 4: kNN QUERIES (k = {0})".format(k))
    print(n)
    print("  {0:<14}{1:>16}{2:>16}".format("method", "avg dist comp", "time (s)"))
    print("  " + "-" * 46)
    print("  {0:<14}{1:>16.2f}{2:>16.4f}".format("Naive",       avg_naive_knn, time_naive_knn))
    print("  {0:<14}{1:>16.2f}{2:>16.4f}".format("Pivot-based", avg_pivot_knn, time_pivot_knn))
    print("  {0:<14}{1:>16.2f}{2:>16.4f}".format("iDistance",   avg_idist_knn, time_idist_knn))


if __name__ == "__main__":
    main()
