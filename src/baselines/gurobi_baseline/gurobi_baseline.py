import gurobipy as gp
from gurobipy import GRB


import sys; sys.path.append("../..")

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


class GurobiBaselineRunner(BaselineRunner):
    def __init__(self, function_name: str, dimensions: int, algorithm: str, displacement: float = None) -> None:
        self.function_name = function_name
        self.dimensions = dimensions
        self.algorithm = algorithm

        if function_name == "Levy":
            self.func = Levy(dimensions, displacement=displacement)
        elif function_name == "Ackley":
            self.func = Ackley(dimensions, displacement=displacement)
        elif function_name == "Dropwave":
            self.func = Dropwave(dimensions, displacement=displacement)
        elif function_name == "SumSquare":
            self.func = SumSquare(dimensions, displacement=displacement)
        elif function_name == "Easom":
            assert dimensions == 2, "Easom is only defined for 2D."
            self.func = Easom(dimensions, displacement=displacement)
        elif function_name == "Michalewicz":
            self.func = Michalewicz(dimensions, displacement=displacement)
        else:
            raise NotImplementedError

        self.model = gp.Model(self.function_name)
        self.x = []
        bounds = self.func.get_default_domain()
        for i in range(dimensions):
            self.x.append(
                self.model.addVar(
                    lb=bounds[i][0],
                    ub=bounds[i][1],
                    vtype=GRB.CONTINUOUS,
                    name="x" + str(i),
                )
            )

        self.model = self.func.encode_to_gurobi(self.model, self.x)

        # for i in range(dimensions):
        #     self.model.addConstr(self.x[i] >= bounds[i][0], name="x" + str(i) + "lower")
        #     self.model.addConstr(self.x[i] <= bounds[i][1], name="x" + str(i) + "upper")

    def run(self) -> dict:
        import io
        from contextlib import redirect_stdout

        with io.StringIO() as buf, redirect_stdout(buf):
            self.model.optimize()
            console_output = buf.getvalue()

        result_dict = GurobiBaselineRunner.process_gurobi_result(
            self.model, self.x, console_output=console_output
        )
        return result_dict

    @staticmethod
    def process_gurobi_result(model, variables, console_output="") -> dict:
        x = [v.X for v in variables]
        obj_val = model.ObjVal
        time_elapsed = model.Runtime

        raw_output = {}
        raw_output["model_status"] = model.Status
        raw_output["MIP_gap"] = model.MIPGap
        raw_output["console_output"] = console_output

        result_dict = {
            "x": x,
            "obj_val": obj_val,
            "time_elapsed": time_elapsed,
            "raw_output": raw_output,
        }
        return result_dict
