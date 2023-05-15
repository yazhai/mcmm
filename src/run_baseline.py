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

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".10"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--func",
    type=str,
    default="SumSquare",
    choices=[
        "Levy",
        "Ackley",
        "Dropwave",
        "SumSquare",
        "Easom",
        "Michalewicz",
        "NeuralNetworkOneLayer",
        "Biggsbi1",
        "Eigenals",
        "Harkerp",
        "Vardim",
        "Watson",
    ],
    help="The function to optimize. Options are: Levy, Ackley, Dropwave, "
    "SumSquare, Easom, Michalewicz, Biggsbi1, Eigenals, Harkerp, Vardim, Watson",
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
        "simulated_annealing",
        "direct",
        "none",
    ],
    help="The algorithm to use. Options are: brute, basinhopping, differential_evolution, "
    "shgo, dual_annealing, simulated_annealing, direct. If solver is gurobi, then this is ignored.",
)
parser.add_argument("--timeout", type=int, default=300, help="The timeout in seconds.")
parser.add_argument("--seed", type=int, default=0, help="The random seed to use.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="output",
    help="The directory to save the output to.",
)
parser.add_argument(
    "--rerun_if_exists",
    default=False,
    action="store_true",
    help="Rerun if output exists.",
)
parser.add_argument(
    "--nn_file_path",
    type=str,
    default=None,
    help="The path to the neural network data file.",
)
parser.add_argument(
    "--displacement",
    type=float,
    default=None,
    help="The displacement in x to use for evaluating functions",
)

args = parser.parse_args()

print("Running with args:")
print(args)


def get_output_fname(args):
    fname = None
    if args.solver == "scipy":
        fname = f"{args.func}_{args.dims}_{args.solver}_{args.algo}_{args.seed}"
    elif args.solver == "gurobi":
        fname = f"{args.func}_{args.dims}_{args.solver}_{args.seed}"
    else:
        raise ValueError(f"Unknown solver {args.solver}")

    if args.func == "NeuralNetworkOneLayer":
        fname += f"-{args.nn_file_path.split('/')[-1].split('.')[0]}"

    return fname


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
record_dir = os.path.join(output_dir, "record")
os.makedirs(record_dir, exist_ok=True)

res_fp = os.path.join(result_dir, "{}.json".format(get_output_fname(args)))
raw_fp = os.path.join(raw_dir, "{}.pickle".format(get_output_fname(args)))
err_fp = os.path.join(err_dir, "{}.txt".format(get_output_fname(args)))
if any([os.path.exists(fp) for fp in [res_fp, raw_fp]]):
    if args.rerun_if_exists:
        print("Output already exists. Rerunning.")
    else:
        print("Output already exists. Skipping.")
        exit(0)

try:
    if args.solver == "scipy":
        assert args.algo != "none", "If solver is scipy, then algo must be specified."
        runner = ScipyBaselineRunner(
            args.func,
            args.dims,
            args.algo,
            timeout=args.timeout,
            nn_file_path=args.nn_file_path,
            displacement=args.displacement,
        )
    elif args.solver == "gurobi":
        runner = GurobiBaselineRunner(
            args.func, args.dims, args.algo, displacement=args.displacement
        )

    result_dict = runner.run()

    if "records" in result_dict:
        filename_records_X = "{}_X.npy".format(get_output_fname(args))
        filename_records_Y = "{}_Y.npy".format(get_output_fname(args))
        np.save(os.path.join(record_dir, filename_records_X), result_dict["records"][0])
        np.save(os.path.join(record_dir, filename_records_Y), result_dict["records"][1])
        del result_dict["records"]

    filename_raw = "{}.pickle".format(get_output_fname(args))
    with open(os.path.join(raw_dir, filename_raw), "wb") as f:
        pickle.dump(result_dict, f)

    filename_result = "{}.json".format(get_output_fname(args))
    with open(os.path.join(result_dir, filename_result), "w") as f:
        del result_dict["raw_output"]
        # result_dict['x'] = result_dict['x'].tolist()
        json.dump(result_dict, f)

except Exception as e:
    stack_trace = traceback.format_exc()
    print(stack_trace)
    filename_err = "{}.txt".format(get_output_fname(args))
    with open(os.path.join(err_dir, filename_err), "w") as f:
        f.write(stack_trace)
