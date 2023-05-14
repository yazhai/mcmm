#!/bin/bash

set -x

eval "$(conda shell.bash hook)"
conda activate mciv

python train_nn_one_layer.py --input_dims 50 --hidden_dims 16 --num_epochs 10000 --batch_size 8000 --learning_rate 0.001 --num_samples 250000 --test_function ackley --seed 0
python train_nn_one_layer.py --input_dims 100 --hidden_dims 16 --num_epochs 20000 --batch_size 8000 --learning_rate 0.001 --num_samples 700000 --test_function ackley --seed 0
python train_nn_one_layer.py --input_dims 50 --hidden_dims 16 --num_epochs 10000 --batch_size 8000 --learning_rate 0.001 --num_samples 250000 --test_function michalewicz --seed 0
python train_nn_one_layer.py --input_dims 100 --hidden_dims 16 --num_epochs 20000 --batch_size 8000 --learning_rate 0.001 --num_samples 700000 --test_function michalewicz --seed 0
# python train_nn_one_layer.py --input_dims 50 --hidden_dims 16 --num_epochs 10000 --batch_size 8000 --learning_rate 0.001 --num_samples 250000 --test_function levy --seed 0
# python train_nn_one_layer.py --input_dims 100 --hidden_dims 16 --num_epochs 10000 --batch_size 8000 --learning_rate 0.001 --num_samples 500000 --test_function levy --seed 0
