import sys, os, string, random
import time, json, fcntl, traceback


from src.test_functions import *
from src.mcir import MCIR


def mcir_test(config_file):
    # ===========================================================================
    # read the json file
    config = json.loads(open(config_file).read())

    # interpret the config file
    fn_class = config.get("objective", {}).get("function", None)

    # define for NN objective function
    if fn_class is None or fn_class == "NeuralNetworkOneLayerTrained":
        fn_class = NeuralNetworkOneLayerTrained
        model = config.get("objective", {}).get("model", None)
        if model is None:
            raise ValueError("NN pretrained model is not specified")
        obj = model
        name = os.path.basename(model).split(".")[0]

    # define for normal objective functions
    else:
        fn_class = eval(fn_class)
        dims = config.get("objective", {}).get("dims", 2)
        obj = dims
        name = fn_class.__name__

    lb = config.get("objective", {}).get("lb", None)
    ub = config.get("objective", {}).get("ub", None)

    obj_args = config.get("objective", {})
    obj_args.pop("function", None)
    obj_args.pop("model", None)
    obj_args.pop("dims", None)
    obj_args.pop("lb", None)
    obj_args.pop("ub", None)
    init_args = config.get("initial", {})
    optimize_args = config.get("optimize", {})

    # ===========================================================================
    # folders for the results
    temp_foler = "temp"
    save_dir = "./benchmark_mcir/"
    os.makedirs(temp_foler, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # ===========================================================================
    # define temperary file saving the print output
    # temp_filename = "".join(
    #     random.choices(string.ascii_lowercase + string.digits, k=40)
    # )
    temp_filename = os.path.basename(config_file).split(".")[0]
    temp_file = os.path.join(temp_foler, temp_filename)

    # Backup original stdout and stderr
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    # Open file for writing
    temp_std = open(temp_file, "w")

    # Redirect stdout and stderr to the file
    sys.stdout = temp_std
    sys.stderr = temp_std

    # ===========================================================================
    # run the optimization
    fn = fn_class(obj, **obj_args)
    if (lb is None) or (ub is None):
        lb, ub = fn.get_default_bounds()

    alg = MCIR(fn, lb, ub, **init_args)
    dims = alg.dims  # retrieve the dims again

    start = time.time()
    try:
        print(f"Start optimization {name} at dim {dims}")
        best_y = alg.optimize(**optimize_args)
    except Exception as e:
        print(f"Optimization {name} at dim {dims} terminated")
        traceback.print_exc()
        best_y = alg.root.y
    end = time.time()
    eclipsed = end - start

    # Reset stdout and stderr back to their original values before your program finishes
    temp_std.close()
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr

    # ===========================================================================
    # save the results

    # get configurations from the object
    max_iter = alg.max_iterations
    n_opt_local = alg.n_opt_local
    node_uct_lb_coeff = alg.node_uct_lb_coeff
    node_uct_box_coeff = alg.node_uct_box_coeff
    node_uct_explore = alg.node_uct_explore
    num_node_expand = alg.num_node_expand
    seed = alg.seed
    time_jit = alg.time_jit

    # get history from the object
    history = alg.history
    total_sample = history.shape[0]
    try:
        first_reach_sample = np.where(history[:, 0] == best_y)[0][0]
    except:
        first_reach_sample = total_sample
    try:
        first_reach_time = first_reach_sample / total_sample * eclipsed
    except:
        first_reach_time = eclipsed

    # std output rename
    save_filename = f"{name}_{dims}d_{temp_filename}"
    save_file = os.path.join(save_dir, save_filename)
    try:
        os.popen("mv {} {}".format(temp_file, save_file))
    except:
        pass

    # dump history file
    history_filename = os.path.join(save_dir, save_filename + ".npy")
    np.save(history_filename, history)

    # save the results
    result_filename = f"00_{name}.txt"
    result_file = os.path.join(save_dir, result_filename)

    with open(result_file, "a+") as fp:
        fcntl.flock(fp, fcntl.LOCK_EX)

        # write title line
        fp.seek(0)
        c = fp.readlines()
        if len(c) == 0:
            fp.write(
                "#fn_name ; dims ; best_y ; seed ; total_time ; first_reach_time ; total_sample ; first_reach_sample; max_iteration ; node_uct_lb_coeff ; node_uct_box_coeff ; node_uct_explore ; num_node_expand ; n_opt_local ; time_jit ; config_filename \n"
            )

        # write results
        fp.write(f"{name} ; ")
        fp.write(f"{dims} ; ")
        fp.write(f"{best_y:.6f} ; ")
        fp.write(f"{seed} ; ")
        fp.write(f"{eclipsed:.4f} ; ")
        fp.write(f"{first_reach_time:.4f} ; ")
        fp.write(f"{total_sample} ; ")
        fp.write(f"{first_reach_sample} ; ")
        fp.write(f"{max_iter} ; ")
        fp.write(f"{node_uct_lb_coeff} ; ")
        fp.write(f"{node_uct_box_coeff} ; ")
        fp.write(f"{node_uct_explore} ; ")
        fp.write(f"{num_node_expand} ; ")
        fp.write(f"{n_opt_local} ; ")
        fp.write(f"{time_jit:.4f} ; ")
        fp.write(f"{os.path.basename(config_file)} \n")

        fcntl.flock(fp, fcntl.LOCK_UN)

    return


if __name__ == "__main__":
    json_file = sys.argv[1]
    mcir_test(json_file)
