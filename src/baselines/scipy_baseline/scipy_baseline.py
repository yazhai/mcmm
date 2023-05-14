import argparse

import time
import numpy as np
from jax import numpy as jnp
import scipy.optimize as opt
import scipy
from scipy.optimize import OptimizeResult
import typing

import sys

from ..test_functions import (
    TestFunction,
    Levy,
    Ackley,
    Dropwave,
    SumSquare,
    Easom,
    Michalewicz,
    NeuralNetworkOneLayerTrained,
)

from ..baseline_runner import BaselineRunner


class ScipyBaselineRunner(BaselineRunner):
    # Default timeout to 1 week
    def __init__(
        self,
        function_name: str,
        dimensions: int,
        algorithm: str,
        timeout: int = 604800,
        nn_file_path: str = None,
    ) -> None:
        self.function_name = function_name
        self.dimensions = dimensions
        self.algorithm = algorithm

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
        elif function_name == "NeuralNetworkOneLayer":
            assert nn_file_path is not None, "Must provide nn_file_path."
            self.func = NeuralNetworkOneLayerTrained(nn_file_path, device="cpu")
        else:
            raise NotImplementedError
    
        self.timeout = timeout

    
    def run(self) -> dict:
        scipy_result = self.run_scipy_optimization(
            self.func,
            self.algorithm,
            timeout=self.timeout,
        )
        result_dict = ScipyBaselineRunner.process_scipy_result(scipy_result)
        return result_dict

    @staticmethod
    def _run_shgo(func, bounds, timeout, return_dict):
        class MinimizeStopperShgo(object):
            def __init__(self, timeout=float("inf"), func=None):
                self.timeout = timeout
                self.start = time.time()
                self.best_x = None
                self.best_obj = float("inf")
                self.timeout_reached = False
                self.func = func

            def __call__(self, xk):
                elapsed = time.time() - self.start
                x = xk

                if self.func is not None:
                    f = np.array(self.func(x))
                else:
                    f = float("inf")

                if f < self.best_obj:
                    self.best_x = x
                    self.best_obj = f
                    return_dict["best_x"] = x
                    return_dict["best_obj"] = f

                if elapsed >= self.timeout:
                    print("Elapsed time: {}. Timeout reached.".format(elapsed))
                    self.timeout_reached = True
                    return True

        result = opt.shgo(
            func, bounds, callback=MinimizeStopperShgo(timeout=timeout, func=func)
        )
        return_dict["opt_result"] = result
        return_dict["timeout"] = False

        return None

    # Default timeout to 1 week
    def run_scipy_optimization(
        self, func: TestFunction, algo: str, timeout=604800
    ) -> typing.Dict:
        if algo == "brute":
            bounds = func.get_default_domain()
            timer_start = time.time()
            result = opt.brute(func, bounds, full_output=True)
            time_elapsed = time.time() - timer_start
            timeout_reached = False
        elif algo == "basinhopping":

            class MinimizeStopperBasinhopping(object):
                def __init__(self, timeout=float("inf")):
                    self.timeout = timeout
                    self.start = time.time()
                    self.best_x = None
                    self.best_obj = float("inf")
                    self.timeout_reached = False

                def __call__(self, x, f, accept):
                    elapsed = time.time() - self.start

                    if f < self.best_obj:
                        self.best_x = x
                        self.best_obj = f

                    if elapsed >= self.timeout:
                        print("Elapsed time: {}. Timeout reached.".format(elapsed))
                        self.timeout_reached = True
                        return True

            bounds = func.get_default_domain()
            start = np.random.uniform(bounds[:, 0], bounds[:, 1])
            timer_start = time.time()
            callback_instance = MinimizeStopperBasinhopping(timeout=timeout)
            result = opt.basinhopping(func, start, callback=callback_instance)
            time_elapsed = time.time() - timer_start
            timeout_reached = callback_instance.timeout_reached
        elif algo == "differential_evolution":

            class MinimizeStopperDifferentialEvolution(object):
                def __init__(self, timeout=float("inf")):
                    self.timeout = timeout
                    self.start = time.time()
                    self.best_x = None
                    self.best_obj = float("inf")
                    self.timeout_reached = False

                def __call__(self, xk, convergence=None):
                    x = xk
                    f = convergence

                    elapsed = time.time() - self.start

                    if f < self.best_obj:
                        self.best_x = x
                        self.best_obj = f

                    if elapsed >= self.timeout:
                        print("Elapsed time: {}. Timeout reached.".format(elapsed))
                        self.timeout_reached = True
                        return True

            bounds = func.get_default_domain()
            timer_start = time.time()
            callback_instance = MinimizeStopperDifferentialEvolution(timeout=timeout)
            result = opt.differential_evolution(
                func, bounds, callback=callback_instance
            )
            time_elapsed = time.time() - timer_start
            timeout_reached = callback_instance.timeout_reached
        elif algo == "shgo":
            import multiprocessing

            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            timeout_reached = False
            return_dict["timeout"] = False
            return_dict["best_x"] = None
            return_dict["best_obj"] = None
            bounds = func.get_default_domain()
            timer_start = time.time()
            p = multiprocessing.Process(
                target=self._run_shgo, args=(func, bounds, timeout, return_dict)
            )
            p.start()
            p.join(timeout=timeout)

            if p.is_alive():
                p.terminate()
                p.join()
                timeout_reached = True

            time_elapsed = time.time() - timer_start
            result = None
            if timeout_reached:
                result = OptimizeResult()
                result.x = return_dict["best_x"]
                result.fun = return_dict["best_obj"]
                result.success = False
                result.message = "Timeout reached"
            else:
                print(return_dict)
                result = return_dict["opt_result"]
            
        elif algo == "dual_annealing":

            class MinimizeStopperDualAnnealing(object):
                def __init__(self, timeout=float("inf")):
                    self.timeout = timeout
                    self.start = time.time()
                    self.best_x = None
                    self.best_obj = float("inf")
                    self.timeout_reached = False

                def __call__(self, x, f, context):
                    elapsed = time.time() - self.start

                    if f < self.best_obj:
                        self.best_x = x
                        self.best_obj = f

                    if elapsed >= self.timeout:
                        print("Elapsed time: {}. Timeout reached.".format(elapsed))
                        self.timeout_reached = True
                        return True

            bounds = func.get_default_domain()
            timer_start = time.time()
            callback_instance = MinimizeStopperDualAnnealing(timeout=timeout)
            result = opt.dual_annealing(func, bounds, callback=callback_instance)
            time_elapsed = time.time() - timer_start
            timeout_reached = callback_instance.timeout_reached
        elif algo == "direct":

            class MinimizeStopperDirect(object):
                def __init__(self, timeout=float("inf"), func=None):
                    self.timeout = timeout
                    self.start = time.time()
                    self.best_x = None
                    self.best_obj = float("inf")
                    self.timeout_reached = False
                    self.func = func

                def __call__(self, xk):
                    elapsed = time.time() - self.start
                    x = xk

                    if self.func is not None:
                        f = np.array(self.func(x))
                    else:
                        f = float("inf")

                    if f < self.best_obj:
                        self.best_x = x
                        self.best_obj = f

                    if elapsed >= self.timeout:
                        print("Elapsed time: {}. Timeout reached.".format(elapsed))
                        self.timeout_reached = True
                        return True

            bounds = func.get_default_domain()
            timer_start = time.time()
            callback_instance = MinimizeStopperDirect(timeout=timeout)
            bounds = [(bounds[i, 0], bounds[i, 1]) for i in range(bounds.shape[0])]
            result = opt.direct(func, bounds, callback=callback_instance)
            time_elapsed = time.time() - timer_start
            timeout_reached = callback_instance.timeout_reached
        else:
            raise NotImplementedError

        return {
            "optimize_result": result,
            "time_elapsed": time_elapsed,
            "timeout_reached": timeout_reached,
        }

    @staticmethod
    def process_scipy_result(scipy_result) -> dict:
        result = scipy_result["optimize_result"]

        if isinstance(result, OptimizeResult):
            x = result.x
            obj_val = result.fun
            raw_output = result
        elif isinstance(result, tuple):
            x = result[0]
            obj_val = result[1]
            raw_output = result
        else:
            assert False, "Unknown result type: {}.".format(type(result))

        if isinstance(x, jnp.ndarray):
            x = np.array(x).tolist()

        if type(x) == np.ndarray:
            x = x.tolist()

        if type(obj_val) == np.ndarray:
            obj_val = obj_val.tolist()

        if isinstance(obj_val, jnp.ndarray):
            obj_val = np.array(obj_val).tolist()

        result_dict = {
            "x": x,
            "obj_val": obj_val,
            "time_elapsed": scipy_result["time_elapsed"],
            "timeout_reached": scipy_result["timeout_reached"],
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
    args = parser.parse_args()
    main(args)
