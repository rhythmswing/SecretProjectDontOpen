from os import stat_result
import numpy as np
from dataclasses import dataclass
from time import time
import copy

@dataclass
class Solution:
    path: list
    distance: float

class ExpRecorder:
    def __init__(self, trace_file=None):
        self.total = 0
        self.nreject = 0

        self.start_time = 0
        self.deepest_level = float('inf')
        self.trace_file = trace_file

        self.current_best = float('inf')

        if trace_file:
            self.fp = open(trace_file, 'w+')
        else:
            self.fp = None


        self.history = []

    def record(self, best_solution):
        if not self.fp:
            return 
        if best_solution < self.current_best:
            self.current_best = best_solution
            self.fp.write("{:.2f}, {}\n".format(
                self.elapsed_time, self.current_best
            ))

            self.history.append([self.elapsed_time, self.current_best])

    def close(self):
        if self.fp:
            self.fp.close()

    def set_level(self, level):
        if level < self.deepest_level:
            self.deepest_level = level 

    def accept(self):
        self.total += 1
    
    def reject(self):
        self.nreject += 1
        self.total += 1
    @property
    def reject_rate(self):
        if self.total == 0:
            return 0
        return self.nreject / self.total

    def set_timer(self, total_time):
        self.start_time = time()
        self.target_time = total_time + self.start_time

    @property
    def elapsed_time(self):
        return time() - self.start_time

    def within_target_time(self):
        if time() > self.target_time:
            return False
        else:
            return True

class SearchTimeout(Exception):
    pass


def reduce_matrix(d_matrix, src=None, dst=None, exclude=False):
    if int(src is None) + int(dst is None) == 1:
        raise ValueError

    def simple_reduce_(d_matrix):
        # reduce row wise
        cost = 0
        reduced_cost = np.min(d_matrix, axis=1)
        d_matrix = (d_matrix.T - reduced_cost).T
        cost += reduced_cost.sum()
        # reduce col wise
        reduced_cost = np.min(d_matrix, axis=0)
        d_matrix -= reduced_cost 
        cost += reduced_cost.sum()
        return d_matrix, cost

    def reduce_matrix_exclude_(d_matrix, src, dst):
        # avoid inplace ops
        d_matrix = copy.deepcopy(d_matrix)
        d_matrix[src, dst] = float('inf')
        return simple_reduce_(d_matrix)

    def reduce_matrix_include_(d_matrix, src, dst):
        d_matrix = copy.deepcopy(d_matrix)
        d_matrix[dst, src] = float('inf')
        d_matrix = np.delete(d_matrix, src, 0)
        d_matrix = np.delete(d_matrix, dst, 1)
        return simple_reduce_(d_matrix)

    if src is None and dst is None:
        return simple_reduce_(d_matrix)
    
    if exclude:
        return reduce_matrix_exclude_(d_matrix, src, dst)
    return reduce_matrix_include_(d_matrix, src, dst)

class TSPAllInOne:

    def __init__(self, path, n_meta=4):
        with open(path, 'r') as f:
            lines = f.readlines()

        assert len(lines) > n_meta and lines[n_meta].strip() == "NODE_COORD_SECTION"
        meta_info = lines[:n_meta]

        for x, y in [x.split(':') for x in meta_info]:
            setattr(self, x.lower().strip(), y.strip())

        def coordfromtext(string):
            string = string.replace("\n", " ")
            data = np.array([float(x) for x in string.split()])
            return data.reshape(-1, 3)
        coords = ''.join(lines[n_meta+1:]).replace("EOF", "")
        coords = coordfromtext(coords)
        self.city_indices = coords[:, 0].astype(int)
        self.city_coords = coords[:, 1:]

        self.city2id = {i: x for x, i in enumerate(self.city_indices)}
 
        self.ncities = len(coords)

        self.dist_mat = np.zeros((self.ncities, self.ncities)) * float("nan")
        for i in self.cities:
            for j in self.cities:
                self.dist(i, j)

        self._first_order_mins = [np.min(x) for x in self.dist_mat]
        self._sec_order_mins = [np.partition(x, 1)[1] for x in self.dist_mat]

        self.solution = Solution(
            path = [None] * (self.ncities),
            distance = float('inf')
        )

    def first_order_min(self, u):
        return self._first_order_mins[self.city2id[u]]

    def second_order_min(self, u):
        return self._sec_order_mins[self.city2id[u]]

    @property
    def cities(self):
        return self.city_indices

    @property
    def city_coord(self, city):
        return self.city_coords[self.city2id[city]]

    def dist(self, u, v):
        uid, vid = [self.city2id[x] for x in [u, v]]
        D = self.dist_mat[uid, vid]
        def _dist(x, y):
            return np.sqrt(((x - y)**2).sum())

        if D != D: # nan
            if uid ==  vid:
                D = float('inf')
            else:
                ucoord = self.city_coords[uid]
                vcoord = self.city_coords[vid]
                D = _dist(ucoord, vcoord)
            self.dist_mat[uid, vid] = D
            self.dist_mat[vid, uid] = self.dist_mat[uid, vid]
            return self.dist_mat[vid, uid]
        return D


import sys
def _tsp(tspdata, cur_bound, cur_weight, level, cur_path, visited):
    if not global_counter.within_target_time():
        raise SearchTimeout
    sys.stdout.write("\rcurrent best solution: {:.2f}, deepest level: {}, reject rate: {:.2f}%, iter/sec: {:.2f}".format(
        tspdata.solution.distance, level, 100.0*global_counter.reject_rate,
        global_counter.total / global_counter.elapsed_time
    ))
    sys.stdout.flush()
    if level == tspdata.ncities:
        if tspdata.dist(cur_path[level-1], cur_path[0]) < float('inf'):
            cur_res = cur_weight + tspdata.dist(cur_path[level-1], cur_path[0])
            if cur_res < tspdata.solution.distance:
                global_counter.record(cur_res)
                tspdata.solution.distance = cur_res
                tspdata.solution.path = cur_path
        return 

    for i in tspdata.cities:
        if visited[i] == False:
            temp = cur_bound
            #print('cur weight')
            cur_weight += tspdata.dist(cur_path[level-1], i)
            #print(cur_weight)
            if level == 1:
                cur_bound -= 0.5 * (tspdata.first_order_min(cur_path[level-1]) + tspdata.first_order_min(i))
            else:
                cur_bound -= 0.5 * (tspdata.second_order_min(cur_path[level-1]) + tspdata.first_order_min(i))

            if cur_bound + cur_weight < tspdata.solution.distance:
                global_counter.accept()
                cur_path[level] = i
                visited[i] = True
                _tsp(tspdata, cur_bound, cur_weight, level+1, cur_path, visited)
            else:
                global_counter.reject()
            cur_weight -= tspdata.dist(cur_path[level-1], i)
            cur_bound = temp

            visited = {x: False for x in tspdata.cities}
            for j in range(level):
                if cur_path[j] is not None:
                    visited[cur_path[j]] = True


def tsp_branch_and_bound(tspdata):

    cur_bound = 0
    cur_path = [None] * (tspdata.ncities+1)
    visited = {x: False for x in tspdata.cities}
    for i in tspdata.cities:
        cur_bound += (tspdata.first_order_min(i) + tspdata.second_order_min(i))
    cur_bound = cur_bound // 2
    start_city = 0
    visited[tspdata.cities[start_city]] = True
    cur_path[0] = tspdata.cities[start_city]

    _tsp(tspdata, cur_bound, 0, 1, cur_path, visited)



data = None
global_counter = None

def solve(args):
    global data, global_counter
    data = TSPAllInOne(args.inst)
    global_counter = ExpRecorder(args.output_path+'.trace')
    global_counter.set_timer(args.time)

    try:
        tsp_branch_and_bound(data)
    except SearchTimeout:
        print()
        print("Exceeded target time. Terminating algorithm.")

    print(data.solution)
    # TODO: output final solution

    global_counter.close()