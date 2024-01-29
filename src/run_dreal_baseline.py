import argparse
import random
import json

import time
import subprocess
import sympy
from sympy.printing.smtlib import smtlib_code
import dreal

# import src.test_functions as test_functions
import test_functions

def generate_input_ctrs(func_domain, sympy_input_vars):
    ctrs = []
    for domain_i, input_var in zip(func_domain, sympy_input_vars):
        domain_i = domain_i.tolist()

        lb = float(domain_i[0])
        ub = float(domain_i[1])
        if float.is_integer(lb):
            lb = int(lb)
        if float.is_integer(ub):
            ub = int(ub)
        ctrs.append(input_var >= lb)
        ctrs.append(input_var <= ub)
    return ctrs


def generate_smt2_code(sympy_expr, input_ctrs, ub_value):
    output_var = sympy.symbols('y')
    func = sympy.Eq(output_var, sympy_expr)
    ub_value = float(ub_value)
    if float.is_integer(ub_value):
        ub_value = int(ub_value)
    output_constr = output_var <= ub_value
    smt_code = sympy.printing.smtlib.smtlib_code([func] + input_ctrs + [output_constr], suffix_expressions=["(check-sat)"])
    return smt_code


def dreal_solve(binary_fp, smt_code, timeout=10):
    tmp_fp = "/tmp/temp.smt2"
    with open(tmp_fp, "w") as f:
        f.write(smt_code)
        
    command = [binary_fp, tmp_fp]
    
    try:
        result = subprocess.run(
            command,
            stdout = subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 'timeout'
    
    output = result.stdout.decode('utf-8')
    
    if 'unsat' in output:
        return 'unsat'
    elif 'delta-sat' in output:
        return 'sat'
    
    assert False, "Invalid output: {}".format(output)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run dReal baseline')
    parser.add_argument('--func', type=str, default='ackley', help='function name')
    parser.add_argument('--dim', type=int, default=0, help='dimension of the function')
    parser.add_argument('--nn_path', type=str, default=None, help='path to the neural network')
    parser.add_argument('--ub', type=float, default=100, help='upper bound of the function')
    parser.add_argument('--binary', type=str, default="/opt/dreal/4.21.06.2/bin/dreal", help='dReal binary file')
    parser.add_argument('--timeout', type=int, default=10, help='timeout in seconds')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    
    func_name = args.func.lower()
    dim = args.dim
    nn_path = args.nn_path
    upper_bound_val = args.ub
    binary_fp = args.binary
    timeout = args.timeout
    seed = args.seed
    
    func = None


    if func_name == 'ackley':
        func = test_functions.Ackley(dim)
    elif func_name == 'michalewicz':
        func = test_functions.Michalewicz(dim)
    elif func_name == 'levy':
        func = test_functions.Levy(dim)
    elif func_name == 'watson':
        func = test_functions.Watson()
    elif func_name == 'harkerp':
        func = test_functions.Harkerp()
    elif func_name == 'biggsbi1':
        func = test_functions.Biggsbi1()
    elif func_name == 'nn':
        func = test_functions.NeuralNetworkOneLayerTrained(nn_path)

    assert func is not None, "Invalid function name: {}".format(func_name)

    print("Function: {}. Dimension: {}. Upper bound: {}. NN path: {}".format(func_name, dim, upper_bound_val, nn_path))

    variables, expression = func.expression()
    sympy_input_vars, sympy_expression = test_functions.convert_to_sympy_expression(expression)
    input_ctrs = generate_input_ctrs(func.get_default_domain(), sympy_input_vars)
    smt2_code = generate_smt2_code(sympy_expression, input_ctrs, upper_bound_val)
    print("Begin solving..")
    start = time.time()
    result = dreal_solve(binary_fp, smt2_code, timeout)
    time_elapsed = time.time() - start
    print("Solved; result: {}. Time used: {}".format(result, time_elapsed))
    print()

    result = {
        "result": result,
        "time": time_elapsed,
        "seed": seed,
        "func_name": func_name,
        "dim": dim,
        "upper_bound_val": upper_bound_val,
        "nn_path": nn_path,
    }
    if nn_path is not None and '/' in nn_path:
        nn_path = nn_path.split('/')[-1]
    result_fp = "results/dreal_{}_{}_{}_{}_{}.json".format(func_name, dim, upper_bound_val, nn_path, seed)
    with open(result_fp, "w") as f:
        json.dump(result, f)
