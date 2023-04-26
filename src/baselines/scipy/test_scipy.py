import argparse

import numpy as np
import scipy.optimize as opt

import sys; sys.path.append('../..')
from test_functions import Levy, Ackley, Dropwave, SumSquare, Easom, Michalewicz



parser = argparse.ArgumentParser()
parser.add_argument('--func', type=str, default='SumSquare',
                    choices=['Levy', 'Ackley', 'Dropwave', 'SumSquare', 'Easom', 'Michalewicz'],
                    help='The function to optimize. Options are: Levy, Ackley, Dropwave, '
                            'SumSquare, Easom, Michalewicz')
parser.add_argument('--dims', type=int, default=2,
                    help='The number of dimensions of the function to optimize.')
parser.add_argument('--algo', type=str, default='brute',
                    choices=['brute', 'basinhopping', 'differential_evolution', 'shgo', 'dual_annealing',
                             'direct'],
                    help='The algorithm to use. Options are: brute, basinhopping, differential_evolution, '
                            'shgo, dual_annealing, direct')
parser.add_argument('--seed', type=int, default=0,
                    help='The random seed to use.')


def main(args):
    func_name = args.func
    dims = args.dims
    algo = args.algo
    seed = args.seed

    np.random.seed(seed)

    if func_name == 'Levy':
        func = Levy(dims)
    elif func_name == 'Ackley':
        func = Ackley(dims)
    elif func_name == 'Dropwave':
        assert dims == 2, 'Dropwave is only defined for 2D.'
        func = Dropwave(dims)
    elif func_name == 'SumSquare':
        func = SumSquare(dims)
    elif func_name == 'Easom':
        assert dims == 2, 'Easom is only defined for 2D.'
        func = Easom(dims)
    elif func_name == 'Michalewicz':
        func = Michalewicz(dims)
    else:
        raise NotImplementedError

    if algo == 'brute':
        bounds = func.get_default_domain()
        result = opt.brute(func, bounds)
    elif algo == 'basinhopping':
        bounds = func.get_default_domain()
        middle = (bounds[:, 0] + bounds[:, 1]) / 2
        result = opt.basinhopping(func, middle)
    elif algo == 'differential_evolution':
        bounds = func.get_default_domain()
        result = opt.differential_evolution(func, bounds)
    elif algo == 'shgo':
        bounds = func.get_default_domain()
        result = opt.shgo(func, bounds)
    elif algo == 'dual_annealing':
        bounds = func.get_default_domain()
        result = opt.dual_annealing(func, bounds)
    elif algo == 'direct':
        bounds = func.get_default_domain()
        result = opt.dual_annealing(func, bounds)
    else:
        raise NotImplementedError
    
    print(result)
    



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)