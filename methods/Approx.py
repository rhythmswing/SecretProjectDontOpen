# This file includes all functions used for the approximation algorithm based on Minimum Spaning Tree

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import time

def prim(g):
    n = g.shape[0]
    tree = g.copy()
    mst = np.inf * np.ones((n, n))
    np.fill_diagonal(tree, np.inf)
    selected = np.zeros(n, dtype=bool)
    selected[0] = True
    for i in range(1, n):
        from_idx, idx = np.divmod(np.argmin(tree[selected]), n)
        from_idx = np.where(selected)[0][from_idx]
        selected[idx] = True
        tree[selected, idx], tree[idx, selected] = np.inf, np.inf
        mst[from_idx, idx], mst[idx, from_idx] = g[from_idx, idx], g[idx, from_idx]
    return mst

def depthFirstSearch(g):
    visited = []
    n = g.shape[0]
    stack = [0]
    while len(stack) != 0:
        curr = stack.pop()
        if curr not in visited:
            visited.append(curr)
            stack.extend(np.where(g[curr] != np.inf)[0])
    return visited

def cycleValue(tour_list, g):
    tour_list
    value = 0
    for i in range(len(tour_list) - 1):
        value += g[tour_list[i], tour_list[i + 1]]
    return value + g[tour_list[-1], tour_list[0]]

def approx(filename, cut_off_time):
    df = pd.read_csv(filename, skiprows=5, header=None, sep=' ')

    start = time.time()
    g = squareform(pdist(df.iloc[:-1, 1:].values))
    np.fill_diagonal(g, np.inf)

    tour_list = depthFirstSearch(prim(g))
    value = cycleValue(tour_list, g)
    end = time.time()

    solution_name = f'{filename[:-4]}_Approx_{cut_off_time}.tsp'
    solution_trace_name = solution_name[:-4] + '.trace'

    print(value, tour_list)

    f = open(solution_name, 'w')
    f.write(f'{int(value)}\n{str(tour_list)[2:-2].replace(" ", "")}')
    f.close()
    
    #df = pd.read_csv()


def solve(args):
    approx(args.inst, args.time)