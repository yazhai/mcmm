# Monte Carlo Tree Search with Interval Bounds and Regional Gradient (MCIR)

## Required packages 

Install required packages as from `requirement.txt` via `conda` and `pip`

## MCIR

MCIR benchmark can be run with command: `python run.py benchmark_settings/xxxx.json`

All source codes are in the `src` directory including test function definitions.



## Baselines

Baselines can be run with the script `src/run_baseline.py`, with the following commandline options:
```
usage: run_baseline.py [-h] [--func {Levy,Ackley,Dropwave,SumSquare,Easom,Michalewicz,NeuralNetworkOneLayer,Biggsbi1,Eigenals,Harkerp,Vardim,Watson}] [--dims DIMS] [--solver {scipy,gurobi}]
                       [--algo {brute,basinhopping,differential_evolution,shgo,dual_annealing,simulated_annealing,direct,none}] [--timeout TIMEOUT] [--seed SEED] [--output_dir OUTPUT_DIR] [--rerun_if_exists]
                       [--nn_file_path NN_FILE_PATH] [--displacement DISPLACEMENT]

optional arguments:
  -h, --help            show this help message and exit
  --func {Levy,Ackley,Dropwave,SumSquare,Easom,Michalewicz,NeuralNetworkOneLayer,Biggsbi1,Eigenals,Harkerp,Vardim,Watson}
                        The function to optimize. Options are: Levy, Ackley, Dropwave, SumSquare, Easom, Michalewicz, Biggsbi1, Eigenals, Harkerp, Vardim, Watson
  --dims DIMS           The number of dimensions of the function to optimize.
  --solver {scipy,gurobi}
                        The solver to use. Options are: scipy, gurobi
  --algo {brute,basinhopping,differential_evolution,shgo,dual_annealing,simulated_annealing,direct,none}
                        The algorithm to use. Options are: brute, basinhopping, differential_evolution, shgo, dual_annealing, simulated_annealing, direct. If solver is gurobi, then this is ignored.
  --timeout TIMEOUT     The timeout in seconds.
  --seed SEED           The random seed to use.
  --output_dir OUTPUT_DIR
                        The directory to save the output to.
  --rerun_if_exists     Rerun if output exists.
  --nn_file_path NN_FILE_PATH
                        The path to the neural network data file.
  --displacement DISPLACEMENT
                        The displacement in x to use for evaluating functions
```

The baseline results obtained in the paper were run with commands located in `src/baselines/baseline_commands` directory. 
* `baseline_commands_syn_functions.txt`: commands to run baselines for synthetic functions
* `baseline_commands_neural_network.txt`: commands to run baselines for artificially constructed neural networks
* `baseline_commands_nlp.txt`: commands to run baselines for realistic functions in [^1]

[^1]: Yash Puranik and Nikolaos V Sahinidis. Bounds tightening based on optimality conditions for370
nonconvex box-constrained optimization. _Journal of Global Optimization_

### Train Artificial Neural Networks

Our simple neural networks that emulate synthetic functions are trained with `src/train_nn_one_layer.py`. The commands to train the neural networks that were used in the paper are provided at the top of the same python script.
