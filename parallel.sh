#!/bin/bash

ls benchmark_settings/*.json | parallel -j$(nproc) "python run.py {} &"

