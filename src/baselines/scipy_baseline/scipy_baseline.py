import argparse

import time
import numpy as np
from jax import numpy as jnp
import scipy.optimize as opt
from scipy.optimize import OptimizeResult
import typing

import sys

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
    NeuralNetworkOneLayerTrained,
    Biggsbi1,
    Eigenals,
    Harkerp,
    Vardim,
    Watson,
)

from ..baseline_runner import BaselineRunner


class TrackerDecorator:
    def __init__(self, func: TestFunction) -> None:
        self.func = func
        self.records_X = []
        self.records_Y = []

    def __getattr__(self, name):
        return getattr(self.func, name)

    def __call__(self, xk):
        y = self.func(xk)
        self.records_X.append(xk)
        self.records_Y.append(y)
        return y

    def clear_records(self):
        self.records_X = []
        self.records_Y = []

    def get_np_records(self):
        return np.array(self.records_X), np.array(self.records_Y)


class ScipyBaselineRunner(BaselineRunner):
    # Default timeout to 1 week
    def __init__(
        self,
        function_name: str,
        dimensions: int,
        algorithm: str,
        timeout: int = 604800,
        nn_file_path: str = None,
        displacement: float = None,
    ) -> None:
        self.function_name = function_name
        self.dimensions = dimensions
        self.algorithm = algorithm

        if function_name == "Levy":
            self.func = TrackerDecorator(Levy(dimensions, displacement=displacement))
        elif function_name == "Ackley":
            self.func = TrackerDecorator(Ackley(dimensions, displacement=displacement))
        elif function_name == "Dropwave":
            self.func = TrackerDecorator(
                Dropwave(dimensions, displacement=displacement)
            )
        elif function_name == "SumSquare":
            self.func = TrackerDecorator(
                SumSquare(dimensions, displacement=displacement)
            )
        elif function_name == "Easom":
            assert dimensions == 2, "Easom is only defined for 2D."
            self.func = TrackerDecorator(Easom(dimensions, displacement=displacement))
        elif function_name == "Michalewicz":
            self.func = TrackerDecorator(
                Michalewicz(dimensions, displacement=displacement)
            )
        elif function_name == "Biggsbi1":
            self.func = TrackerDecorator(
                Biggsbi1(dimensions, displacement=displacement)
            )
        elif function_name == "Eigenals":
            self.func = TrackerDecorator(
                Eigenals(dimensions, displacement=displacement)
            )
        elif function_name == "Harkerp":
            self.func = TrackerDecorator(Harkerp(dimensions, displacement=displacement))
        elif function_name == "Vardim":
            self.func = TrackerDecorator(Vardim(dimensions, displacement=displacement))
        elif function_name == "Watson":
            self.func = TrackerDecorator(Watson(dimensions, displacement=displacement))
        elif function_name == "NeuralNetworkOneLayer":
            assert nn_file_path is not None, "Must provide nn_file_path."
            self.func = TrackerDecorator(NeuralNetworkOneLayerTrained(nn_file_path, device="cpu"))
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
            result = opt.basinhopping(
                func, start, niter=1000, niter_success=1000, callback=callback_instance
            )
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
                func, bounds, maxiter=1000000, callback=callback_instance
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

        elif algo == "simulated_annealing":

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
            result = opt.dual_annealing(
                func, bounds, no_local_search=True, callback=callback_instance
            )
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
            result = opt.direct(
                func,
                bounds,
                maxiter=100000,
                locally_biased=False,
                callback=callback_instance,
            )
            time_elapsed = time.time() - timer_start
            timeout_reached = callback_instance.timeout_reached
        else:
            raise NotImplementedError

        return {
            "optimize_result": result,
            "time_elapsed": time_elapsed,
            "timeout_reached": timeout_reached,
            "records": func.get_np_records(),
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
            "records": scipy_result["records"],
        }
        return result_dict
