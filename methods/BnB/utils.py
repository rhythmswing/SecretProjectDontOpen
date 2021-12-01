
from os import stat_result
import numpy as np
from dataclasses import dataclass
from time import time
import copy
from methods.BnB.utils import *


@dataclass
class Solution:
    """ data class containing a solution.
    """
    path: list
    distance: float
    def __init__(self, path, distance):
        self.distance = distance
        # make sure path is independently stored 
        self.path = copy.deepcopy(path)

class ExpRecorder:
    """ Maintaining global variables and logs.
    """
    def __init__(self, output_file=None):
        self.total = 0
        self.nreject = 0

        self.start_time = 0
        self.deepest_level = float('inf')

        self.current_best = Solution([], float('inf'))

        if output_file:
            self.trace_file = output_file + '.trace'
            self.sol_file = output_file + '.sol'


        self.history = []

    def record(self, best_solution):
        """ Record a best solution to the output trace file.
        """
        if best_solution.distance < self.current_best.distance:
            self.current_best = best_solution
            self.history.append([self.elapsed_time, self.current_best])

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

    def write_trace(self):
        with open(self.trace_file, 'w+') as f:
            for timestep, sol in self.history:
                f.write("{:.2f}, {}\n".format(
                    timestep, int(sol.distance)
                ))

    def write_solution(self):
        with open(self.sol_file, 'w+') as f:
            f.write("{}\n{}\n".format(
                int(self.current_best.distance),
                ",".join(map(str, self.current_best.path))
            ))

class SearchTimeout(Exception):
    # Dummy exception class for calling timeout 
    pass


def reduce_matrix(d_matrix, src=None, dst=None, exclude=False):
    """ Reduce a distance matrix for TSP branch and bound.

    Args:
        d_matrix (np.array): distance matrix.
        src (int, optional): source node. Defaults to None.
        dst ([type], optional): destination node. Defaults to None.
        exclude (bool, optional): if the computation of lower bound includes or excludes node. Defaults to False.

    Returns:
        [np.array, float]: reduced matrix and cost.
    """
    if int(src is None) + int(dst is None) == 1:
        raise ValueError

    def simple_reduce_(d_matrix):
        # reduce row wise
        cost = 0
        reduced_cost = np.min(d_matrix, axis=1)
        reduced_cost[reduced_cost==float('inf')]=0
        d_matrix = (d_matrix.T - reduced_cost).T
        # remove nan caused by min([inf, inf])
        cost += reduced_cost.sum()
        # reduce col wise
        reduced_cost = np.min(d_matrix, axis=0)
        reduced_cost[reduced_cost==float('inf')]=0
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
        d_matrix[src, :] = float('inf')
        d_matrix[:, dst] = float('inf')
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
