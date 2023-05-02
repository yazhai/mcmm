import argparse

import time
import numpy as np
import scipy.optimize as opt
from scipy.optimize import OptimizeResult

import sys

sys.path.append("../..")
from test_functions import (
    TestFunction,
    Levy,
    Ackley,
    Dropwave,
    SumSquare,
    Easom,
    Michalewicz,
)

from ..baseline_runner import BaselineRunner


parser = argparse.ArgumentParser()
parser.add_argument(
    "--func",
    type=str,
    default="SumSquare",
    choices=["Levy", "Ackley", "Dropwave", "SumSquare", "Easom", "Michalewicz"],
    help="The function to optimize. Options are: Levy, Ackley, Dropwave, "
    "SumSquare, Easom, Michalewicz",
)
parser.add_argument(
    "--dims",
    type=int,
    default=2,
    help="The number of dimensions of the function to optimize.",
)
parser.add_argument(
    "--algo",
    type=str,
    default="brute",
    choices=[
        "brute",
        "basinhopping",
        "differential_evolution",
        "shgo",
        "dual_annealing",
        "direct",
    ],
    help="The algorithm to use. Options are: brute, basinhopping, differential_evolution, "
    "shgo, dual_annealing, direct",
)
parser.add_argument("--seed", type=int, default=0, help="The random seed to use.")


class ScipyBaselineRunner(BaselineRunner):
    def __init__(
        self, function_name: str, dimensions: int, algorithm: str, seed: int
    ) -> None:
        self.function_name = function_name
        self.dimensions = dimensions
        self.algorithm = algorithm
        self.seed = seed

        if function_name == "Levy":
            self.func = Levy(dimensions)
        elif function_name == "Ackley":
            self.func = Ackley(dimensions)
        elif function_name == "Dropwave":
            self.func = Dropwave(dimensions)
        elif function_name == "SumSquare":
            self.func = SumSquare(dimensions)
        elif function_name == "Easom":
            assert dimensions == 2, "Easom is only defined for 2D."
            self.func = Easom(dimensions)
        elif function_name == "Michalewicz":
            self.func = Michalewicz(dimensions)
        else:
            raise NotImplementedError

    def run(self) -> dict:
        result, time_elapsed = self.run_scipy_optimization(
            self.func, self.algorithm, self.seed
        )
        result_dict = ScipyBaselineRunner.process_scipy_result(
            result, time_elapsed=time_elapsed
        )
        return result_dict

    def run_scipy_optimization(
        self, func: TestFunction, algo: str, seed: int
    ) -> OptimizeResult:
        if algo == "brute":
            bounds = func.get_default_domain()
            timer_start = time.time()
            result = opt.brute(func, bounds)
            time_elapsed = time.time() - timer_start
        elif algo == "basinhopping":
            bounds = func.get_default_domain()
            start = np.random.uniform(bounds[:, 0], bounds[:, 1])
            timer_start = time.time()
            result = opt.basinhopping(func, start)
            time_elapsed = time.time() - timer_start
        elif algo == "differential_evolution":
            bounds = func.get_default_domain()
            timer_start = time.time()
            result = opt.differential_evolution(func, bounds)
            time_elapsed = time.time() - timer_start
        elif algo == "shgo":
            bounds = func.get_default_domain()
            timer_start = time.time()
            result = opt.shgo(func, bounds)
            time_elapsed = time.time() - timer_start
        elif algo == "dual_annealing":
            bounds = func.get_default_domain()
            timer_start = time.time()
            result = opt.dual_annealing(func, bounds)
            time_elapsed = time.time() - timer_start
        elif algo == "direct":
            bounds = func.get_default_domain()
            timer_start = time.time()
            result = opt.dual_annealing(func, bounds)
            time_elapsed = time.time() - timer_start
        else:
            raise NotImplementedError

        return result, time_elapsed

    @staticmethod
    def process_scipy_result(result, time_elapsed=None) -> dict:
        x = result.x
        obj_val = result.fun
        raw_output = result
        result_dict = {
            "x": x,
            "obj_val": obj_val,
            "time_elapsed": time_elapsed,
            "raw_output": raw_output,
        }
        return result_dict


def run_scipy_optimization(func: TestFunction, algo: str, seed: int) -> OptimizeResult:
    np.random.seed(seed)

    if algo == "brute":
        bounds = func.get_default_domain()
        result = opt.brute(func, bounds)
    elif algo == "basinhopping":
        bounds = func.get_default_domain()
        start = np.random.uniform(bounds[:, 0], bounds[:, 1])
        print(start)
        result = opt.basinhopping(func, start)
    elif algo == "differential_evolution":
        bounds = func.get_default_domain()
        result = opt.differential_evolution(func, bounds)
    elif algo == "shgo":
        bounds = func.get_default_domain()
        result = opt.shgo(func, bounds)
    elif algo == "dual_annealing":
        bounds = func.get_default_domain()
        result = opt.dual_annealing(func, bounds)
    elif algo == "direct":
        bounds = func.get_default_domain()
        result = opt.dual_annealing(func, bounds)
    else:
        raise NotImplementedError

    return result


def main(args):
    func_name = args.func
    dims = args.dims
    algo = args.algo
    seed = args.seed

    np.random.seed(seed)

    if func_name == "Levy":
        func = Levy(dims)
    elif func_name == "Ackley":
        func = Ackley(dims)
    elif func_name == "Dropwave":
        assert dims == 2, "Dropwave is only defined for 2D."
        func = Dropwave(dims)
    elif func_name == "SumSquare":
        func = SumSquare(dims)
    elif func_name == "Easom":
        assert dims == 2, "Easom is only defined for 2D."
        func = Easom(dims)
    elif func_name == "Michalewicz":
        func = Michalewicz(dims)
    else:
        raise NotImplementedError

    if algo == "brute":
        bounds = func.get_default_domain()
        result = opt.brute(func, bounds)
    elif algo == "basinhopping":
        bounds = func.get_default_domain()
        start = np.random.uniform(bounds[:, 0], bounds[:, 1])
        print(start)
        result = opt.basinhopping(func, start)
    elif algo == "differential_evolution":
        bounds = func.get_default_domain()
        result = opt.differential_evolution(func, bounds)
    elif algo == "shgo":
        bounds = func.get_default_domain()
        result = opt.shgo(func, bounds)
    elif algo == "dual_annealing":
        bounds = func.get_default_domain()
        result = opt.dual_annealing(func, bounds)
    elif algo == "direct":
        bounds = func.get_default_domain()
        result = opt.dual_annealing(func, bounds)
    else:
        raise NotImplementedError

    print(result)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
