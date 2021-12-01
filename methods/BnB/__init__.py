
data = None
global_counter = None

from .utils import *

def solve(args):
    global data, global_counter
    data = TSPAllInOne(args.inst)
    global_counter = ExpRecorder(args.output_path)
    global_counter.set_timer(args.time)

    if args.bnb_bound == 'reduce_matrix':
        from methods.BnB.reduce_matrix import tsp_branch_and_bound
    elif args.bnb_bound == 'smallest_edge':
        from methods.BnB.smallest_edge import tsp_branch_and_bound


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
