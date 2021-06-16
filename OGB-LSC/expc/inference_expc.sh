# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env bash

dataset=$1
checkpoint=$2
output=$3

config="./src/inference_tool/ogbg-pcqm4m.json"

python ./src/inference_tool/main_inference.py $config  $dataset  $checkpoint  $output



