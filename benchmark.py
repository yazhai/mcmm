import time

from src.test_functions import *
from src.mciv import MCIV


dims_all = [2, 10, 50, 100, 200]
max_iterations = [20, 40, 100, 200]
node_expansions = [10, 10, 10, 10]

fn_all = [Ackley, Levy, Michalewicz]

export_log = "./benchmarks/result_summary.txt"

with open(export_log, "w") as f:
    f.write("function ; result ; time\n")


for ii in range(len(dims_all)):
    if ii < 2:
        continue
    for rep in range(5):
        dims = dims_all[ii]
        max_iter = max_iterations[ii]
        num_node_expand = node_expansions[ii]

        lb = -10 * np.ones(dims)
        ub = 10 * np.ones(dims)

        for fn_class in fn_all:
            fn = fn_class(dims=dims)
            var_exp = None
            fn_exp = None
            try:
                var_exp, fn_exp = fn.expression()
            except:
                pass

            name = fn.__class__.__name__ + "_dim" + str(dims) + "_run" + str(rep)
            print(f"=== Start running with {name} \n")

            alg = MCIV(
                fn=fn,
                lb=lb,
                ub=ub,
                function_variables=var_exp,
                function_expression=fn_exp,
            )

            start = time.time()
            try:
                res = alg.optimize(
                    verbose=3,
                    max_iterations=max_iter,
                    num_node_expand=num_node_expand,
                    node_uct_lb_coeff=1.0,
                    node_uct_explore=1.0,
                )
                status = "success"
            except:
                res = alg.root.y
                status = "failed"
            end = time.time()
            eclipse = end - start
            trajectory = alg.history

            np.save(f"./benchmarks/{name}.npy", trajectory)
            with open(export_log, "a") as f:
                f.write(f"{name} ; {res:.4f} ; {eclipse:.2f} ; {status} \n")
