import sys
import time

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-inst', type=str, required=True
    )

    parser.add_argument(
        '-alg', choices=['BnB', 'Approx', 'LS1', 'LS2'], required=True
    )

    parser.add_argument(
        '-time', type=int, required=True
    )

    parser.add_argument(
        '-seed', type=int, default=None
    )

    args = parser.parse_args()
    return args


def dispatch_args(args):

    def set_seed(seed):
        import random
        import numpy
        random.seed(seed)
        numpy.random.seed(seed)

    def generate_output_path(args):
        import os
        inst = args.inst
        city_file = os.path.split(inst)[1]
        city_name = os.path.splitext(city_file)[0]
        file_name = "{}_{}_{}".format(
            city_name, args.alg, args.time)
        if args.seed is not None:
            file_name += "_" + str(args.seed)
        return file_name
    
    output_path = generate_output_path(args)
    args.output_path = output_path
    set_seed(args.seed)

    import importlib
    try:
        method_module = importlib.import_module('methods.{}'.format(args.alg))
    except:
        print("Algorithm {} is not found in methods/. Check code.".format(args.alg))

    method_module.solve(args)



if __name__ == '__main__':
    args = parse_arguments()
    dispatch_args(args)

