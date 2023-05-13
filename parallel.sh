#!/bin/bash

files=$(ls ./benchmark_settings/*.json)

echo "$files" | parallel "python run.py {} "

