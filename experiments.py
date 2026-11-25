# DAFERMOU CHRISTINA DANAE - 5206
# VALERAS CHRISTOS - 5182

# Meros 5: peirama gia diafora eps kai k kai ta plots

import sys
import time
import matplotlib
matplotlib.use("Agg")  # gia na min thelei display
import matplotlib.pyplot as plt

#  import oles tis sunartiseis apo to kurio arxeio
import assignment3 as a3


def run_experiments(data_file, query_file, numpivots):
    # diavasma dedomenwn mia fora
    print("Loading data and queries...")
    D = a3.load_data(data_file)
    Q = a3.load_data(query_file)
    print("Computing pivots...")
    pivots, distances = a3.compute_pivots(D, numpivots)
    a3.distances_global = distances
    idist_array, maxd_p, maxd, nearest_pivot = a3.build_idistance(distances, numpivots)
    partition_objs = [[] for _ in range(numpivots)]
    for i in range(len(D)):
        partition_objs[nearest_pivot[i]].append(i)

    # ===== RANGE QUERIES gia eps = 0.1, 0.2, 0.4, 0.8 =====
    eps_values = [0.1, 0.2, 0.4, 0.8]
    range_naive_times = []
    range_pivot_times = []
    range_idist_times = []
    range_naive_comps = []
    range_pivot_comps = []
    range_idist_comps = []

    for eps in eps_values:
        print("---- eps =", eps, "----")

        # Naive
        a3.dist_count = 0
        t0 = time.time()
        for q in Q:
            a3.range_naive(D, q, eps)
        t1 = time.time()
        range_naive_times.append(t1 - t0)
        range_naive_comps.append(a3.dist_count / len(Q))

        # Pivot
        a3.dist_count = 0
        t0 = time.time()
        for q in Q:
            a3.range_pivot(D, q, eps, pivots, distances)
        t1 = time.time()
        range_pivot_times.append(t1 - t0)
        range_pivot_comps.append(a3.dist_count / len(Q))

        # iDistance
        a3.dist_count = 0
        t0 = time.time()
        for q in Q:
            a3.range_idistance(D, q, eps, pivots, idist_array, maxd, nearest_pivot)
        t1 = time.time()
        range_idist_times.append(t1 - t0)
        range_idist_comps.append(a3.dist_count / len(Q))

        print("  Naive:    time=%.4f comps=%.2f" % (range_naive_times[-1], range_naive_comps[-1]))
        print("  Pivot:    time=%.4f comps=%.2f" % (range_pivot_times[-1], range_pivot_comps[-1]))
        print("  iDist:    time=%.4f comps=%.2f" % (range_idist_times[-1], range_idist_comps[-1]))

    # PLOT 1: time vs eps
    plt.figure(figsize=(8, 5))
    plt.plot(eps_values, range_naive_times, marker='o', label='Naive')
    plt.plot(eps_values, range_pivot_times, marker='s', label='Pivot-based')
    plt.plot(eps_values, range_idist_times, marker='^', label='iDistance')
    plt.xlabel("epsilon")
    plt.ylabel("Total time (sec)")
    plt.title("Range queries: total time vs epsilon  (numpivots=%d)" % numpivots)
    plt.legend()
    plt.grid(True)
    plt.savefig("range_time.png", dpi=120, bbox_inches='tight')
    plt.close()

    # PLOT 2: distance comps vs eps
    plt.figure(figsize=(8, 5))
    plt.plot(eps_values, range_naive_comps, marker='o', label='Naive')
    plt.plot(eps_values, range_pivot_comps, marker='s', label='Pivot-based')
    plt.plot(eps_values, range_idist_comps, marker='^', label='iDistance')
    plt.xlabel("epsilon")
    plt.ylabel("avg distance computations per query")
    plt.title("Range queries: dist. computations vs epsilon  (numpivots=%d)" % numpivots)
    plt.legend()
    plt.grid(True)
    plt.savefig("range_comps.png", dpi=120, bbox_inches='tight')
    plt.close()

    # ===== kNN QUERIES gia k = 1,5,10,50,100 =====
    k_values = [1, 5, 10, 50, 100]
    knn_naive_times = []
    knn_pivot_times = []
    knn_idist_times = []
    knn_naive_comps = []
    knn_pivot_comps = []
    knn_idist_comps = []

    for k in k_values:
        print("---- k =", k, "----")

        # Naive
        a3.dist_count = 0
        t0 = time.time()
        for q in Q:
            a3.knn_naive(D, q, k)
        t1 = time.time()
        knn_naive_times.append(t1 - t0)
        knn_naive_comps.append(a3.dist_count / len(Q))

        # Pivot
        a3.dist_count = 0
        t0 = time.time()
        for q in Q:
            a3.knn_pivot(D, q, k, pivots, distances)
        t1 = time.time()
        knn_pivot_times.append(t1 - t0)
        knn_pivot_comps.append(a3.dist_count / len(Q))

        # iDistance
        a3.dist_count = 0
        t0 = time.time()
        for q in Q:
            a3.knn_idistance(D, q, k, pivots, idist_array, maxd, nearest_pivot, partition_objs)
        t1 = time.time()
        knn_idist_times.append(t1 - t0)
        knn_idist_comps.append(a3.dist_count / len(Q))

        print("  Naive:    time=%.4f comps=%.2f" % (knn_naive_times[-1], knn_naive_comps[-1]))
        print("  Pivot:    time=%.4f comps=%.2f" % (knn_pivot_times[-1], knn_pivot_comps[-1]))
        print("  iDist:    time=%.4f comps=%.2f" % (knn_idist_times[-1], knn_idist_comps[-1]))

    # PLOT 3: time vs k
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, knn_naive_times, marker='o', label='Naive')
    plt.plot(k_values, knn_pivot_times, marker='s', label='Pivot-based')
    plt.plot(k_values, knn_idist_times, marker='^', label='iDistance')
    plt.xlabel("k")
    plt.ylabel("Total time (sec)")
    plt.title("kNN queries: total time vs k  (numpivots=%d)" % numpivots)
    plt.legend()
    plt.grid(True)
    plt.savefig("knn_time.png", dpi=120, bbox_inches='tight')
    plt.close()

    # PLOT 4: distance comps vs k
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, knn_naive_comps, marker='o', label='Naive')
    plt.plot(k_values, knn_pivot_comps, marker='s', label='Pivot-based')
    plt.plot(k_values, knn_idist_comps, marker='^', label='iDistance')
    plt.xlabel("k")
    plt.ylabel("avg distance computations per query")
    plt.title("kNN queries: dist. computations vs k  (numpivots=%d)" % numpivots)
    plt.legend()
    plt.grid(True)
    plt.savefig("knn_comps.png", dpi=120, bbox_inches='tight')
    plt.close()

    print("\nDONE - plots saved: range_time.png, range_comps.png, knn_time.png, knn_comps.png")

    # epistrefw kai ta dedomena gia na ta vazoume sto report
    return {
        "eps_values": eps_values,
        "range_naive_times": range_naive_times,
        "range_pivot_times": range_pivot_times,
        "range_idist_times": range_idist_times,
        "range_naive_comps": range_naive_comps,
        "range_pivot_comps": range_pivot_comps,
        "range_idist_comps": range_idist_comps,
        "k_values": k_values,
        "knn_naive_times": knn_naive_times,
        "knn_pivot_times": knn_pivot_times,
        "knn_idist_times": knn_idist_times,
        "knn_naive_comps": knn_naive_comps,
        "knn_pivot_comps": knn_pivot_comps,
        "knn_idist_comps": knn_idist_comps,
    }


if __name__ == "__main__":
    numpivots = 10
    data_file = "data10K10.txt"
    query_file = "queries10.txt"
    if len(sys.argv) >= 2:
        numpivots = int(sys.argv[1])
    if len(sys.argv) >= 3:
        data_file = sys.argv[2]
    if len(sys.argv) >= 4:
        query_file = sys.argv[3]
    run_experiments(data_file, query_file, numpivots)
