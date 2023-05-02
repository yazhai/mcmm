import argparse
import random
import numpy as np
import os
import pickle
import json
import traceback
from datetime import datetime

from baselines.gurobi_baseline.gurobi_baseline import GurobiBaselineRunner
from baselines.scipy_baseline.scipy_baseline import ScipyBaselineRunner


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
    "--solver",
    type=str,
    default="scipy",
    choices=["scipy", "gurobi"],
    help="The solver to use. Options are: scipy, gurobi",
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
        "none",
    ],
    help="The algorithm to use. Options are: brute, basinhopping, differential_evolution, "
    "shgo, dual_annealing, direct. If solver is gurobi, then this is ignored.",
)
parser.add_argument("--timeout", type=int, default=300, help="The timeout in seconds.")
parser.add_argument("--seed", type=int, default=0, help="The random seed to use.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="output",
    help="The directory to save the output to.",
)


args = parser.parse_args()

print("Running with args:")
print(args)

def get_output_fname(args):
    if args.solver == "scipy":
        return f"{args.func}_{args.dims}_{args.solver}_{args.algo}_{args.seed}"
    elif args.solver == "gurobi":
        return f"{args.func}_{args.dims}_{args.solver}_{args.seed}"
    else:
        raise ValueError(f"Unknown solver {args.solver}")


seed = args.seed
random.seed(seed)
np.random.seed(seed)

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
print(f"Saving output to {output_dir}")

result_dir = os.path.join(output_dir, "results")
os.makedirs(result_dir, exist_ok=True)
raw_dir = os.path.join(output_dir, "raw")
os.makedirs(raw_dir, exist_ok=True)
err_dir = os.path.join(output_dir, "err")
os.makedirs(err_dir, exist_ok=True)

try:
    if args.solver == "scipy":
        assert args.algo != "none", "If solver is scipy, then algo must be specified."
        runner = ScipyBaselineRunner(args.func, args.dims, args.algo, timeout=args.timeout)
    elif args.solver == "gurobi":
        runner = GurobiBaselineRunner(args.func, args.dims, args.algo)

    result_dict = runner.run()

    filename_raw = "{}.pickle".format(get_output_fname(args))
    with open(os.path.join(raw_dir, filename_raw), "wb") as f:
        pickle.dump(result_dict, f)

    filename_result = "{}.json".format(get_output_fname(args))
    with open(os.path.join(result_dir, filename_result), "w") as f:
        del result_dict['raw_output']
        # result_dict['x'] = result_dict['x'].tolist()
        json.dump(result_dict, f)

except Exception as e:
    stack_trace = traceback.format_exc()
    print(stack_trace)
    filename_err = "{}.txt".format(get_output_fname(args))
    with open(os.path.join(err_dir, filename_err), "w") as f:
        f.write(stack_trace)

