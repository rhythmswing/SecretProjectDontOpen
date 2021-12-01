from os import stat_result
import numpy as np
from dataclasses import dataclass
from time import time
import copy
from methods.BnB.utils import *

import sys
from . import global_counter, data
def _tsp(tspdata, cur_bound, cur_weight, level, cur_path, visited, dist_mat=None):
    if not global_counter.within_target_time():
        raise SearchTimeout
    sys.stdout.write("\rcurrent best solution: {:.2f}, deepest level: {}, reject rate: {:.2f}%, iter/sec: {:.2f}".format(
        tspdata.solution.distance, level, 100.0*global_counter.reject_rate,
        global_counter.total / global_counter.elapsed_time
    ))
    sys.stdout.flush()
    if level == tspdata.ncities:
        cur_res = cur_weight + tspdata.dist(cur_path[level-1], cur_path[0])
        if cur_res < tspdata.solution.distance:
            tspdata.solution = Solution(cur_path, cur_res)
            global_counter.record(tspdata.solution)
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
    cur_path = [None] * (tspdata.ncities)
    visited = {x: False for x in tspdata.cities}
    for i in tspdata.cities:
        cur_bound += (tspdata.first_order_min(i) + tspdata.second_order_min(i))
    cur_bound = cur_bound // 2
    start_city = 0
    visited[tspdata.cities[start_city]] = True
    cur_path[0] = tspdata.cities[start_city]

    _tsp(tspdata, cur_bound, 0, 1, cur_path, visited)


def solve(args):
    data = TSPAllInOne(args.inst)
    global_counter = ExpRecorder(args.output_path)
    global_counter.set_timer(args.time)

    try:
        tsp_branch_and_bound(data)
    except SearchTimeout:
        print()
        print("Exceeded target time. Terminating algorithm.")
    except KeyboardInterrupt:
        print()
        print("Keyboard interrupt accepted. Attempting graceful exit..")
        global_counter.sol_file += "_int_elapsed_{}".format(global_counter.elapsed_time)
        global_counter.trace_file += "_int_elapsed_{}".format(global_counter.elapsed_time)

    print()
    print("Total run time: {:.2f}, reject rate: {:.2f}".format(global_counter.elapsed_time,
        global_counter.reject_rate))
    print("Best solution: ")
    print(data.solution)
    # TODO: output final solution
    global_counter.write_solution()
    global_counter.write_trace()
