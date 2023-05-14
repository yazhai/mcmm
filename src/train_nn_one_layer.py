from test_functions import Levy, Ackley, Dropwave, SumSquare, Michalewicz, NeuralNetworkOneLayer

import random
import torch
import numpy as np

from tqdm import tqdm

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".20"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_dims", type=int, default=2)
parser.add_argument("--hidden_dims", type=int, default=16)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--num_samples", type=int, default=10_000)
parser.add_argument("--test_function", type=str, default="ackley")
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

print()
print("Running with args:")
print(args)
print()
# Example usage:
# python train_nn_one_layer.py --input_dims 2 --hidden_dims 16 --num_epochs 1000 --batch_size 1000 --learning_rate 0.001 --num_samples 50000 --test_function ackley --seed 0
# python train_nn_one_layer.py --input_dims 10 --hidden_dims 16 --num_epochs 5000 --batch_size 1000 --learning_rate 0.001 --num_samples 250000 --test_function ackley --seed 0
# python train_nn_one_layer.py --input_dims 2 --hidden_dims 16 --num_epochs 1000 --batch_size 1000 --learning_rate 0.001 --num_samples 50000 --test_function michalewicz --seed 0
# python train_nn_one_layer.py --input_dims 10 --hidden_dims 16 --num_epochs 5000 --batch_size 8000 --learning_rate 0.001 --num_samples 250000 --test_function michalewicz --seed 0
# python train_nn_one_layer.py --input_dims 2 --hidden_dims 16 --num_epochs 1000 --batch_size 1000 --learning_rate 0.001 --num_samples 50000 --test_function levy --seed 0
# python train_nn_one_layer.py --input_dims 10 --hidden_dims 16 --num_epochs 5000 --batch_size 5000 --learning_rate 0.001 --num_samples 250000 --test_function levy --seed 0


# Set random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

input_dims = args.input_dims
hidden_dims = args.hidden_dims

# Create a test function
if args.test_function.lower() == "levy":
    gt_func = Levy(dims=input_dims)
elif args.test_function.lower() == "ackley":
    gt_func = Ackley(dims=input_dims)
elif args.test_function.lower() == "dropwave":
    gt_func = Dropwave(dims=input_dims)
elif args.test_function.lower() == "sumsquare":
    gt_func = SumSquare(dims=input_dims)
elif args.test_function.lower() == "michalewicz":
    gt_func = Michalewicz(dims=input_dims)
else:
    raise ValueError(f"Unknown test function: {args.test_function}")

bounds = gt_func.get_default_domain()

# Create a neural network with one hidden layer
nn = NeuralNetworkOneLayer(dims=input_dims, domain=bounds, hidden_dims=hidden_dims)

# Create a dataset
print(f"Creating a dataset with {args.num_samples} samples")
num_smaples = args.num_samples
xs = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(num_smaples, input_dims))
ys = []
for x in tqdm(xs):
    ys.append([gt_func(x)])
ys = np.array(ys)

# Train the neural network
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate

criteria = torch.nn.MSELoss()
optimizer = torch.optim.Adam(nn.model.parameters(), lr=learning_rate)

if args.device is not None:
    device = torch.device(args.device)
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nn.model.to(device)
nn.model.train()

print(f"Training the neural network for {num_epochs} epochs")
for epoch in tqdm(range(num_epochs)):
    shuffled_indices = np.random.permutation(num_smaples)
    xs_shuffled = xs[shuffled_indices]
    ys_shuffled = ys[shuffled_indices]

    for i in range(0, num_smaples, batch_size):
        x = torch.FloatTensor(xs_shuffled[i:i+batch_size])
        y = torch.FloatTensor(ys_shuffled[i:i+batch_size])
        
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = nn.model(x)
        loss = criteria(y_pred, y)
        loss.backward()
        optimizer.step()

# Evalute the neural network
nn.model.eval()
losses = []
for i in range(0, num_smaples, batch_size):
    x = torch.FloatTensor(xs[i:i+batch_size])
    y = torch.FloatTensor(ys[i:i+batch_size])
    
    x = x.to(device)
    y = y.to(device)
    
    with torch.no_grad():
        y_pred = nn.model(x)
    loss = criteria(y_pred, y)
    losses.append(loss.item())

print(f"Mean loss: {np.mean(losses)}")

# Save the model
print("Saving the model")
model_info = {
    "input_dims": input_dims,
    "hidden_dims": hidden_dims,
    "test_function": args.test_function,
    "bounds": bounds,
    "state_dict": nn.model.state_dict(),
}
torch.save(model_info, f"nn_models/nn_one_layer_{args.test_function}_{args.input_dims}_{args.hidden_dims}.pt")