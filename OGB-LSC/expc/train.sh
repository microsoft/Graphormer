# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8

output_dir="./pcqm4m_results/"
config_file="./src/ogbg-pcqm4m.json"

time_stamp=`date '+%s'`
commit_id=`git rev-parse HEAD`
std_file=${output_dir}"stdout/"${time_stamp}_${commit_id}".txt"

mkdir -p $output_dir"stdout/"

nohup python -u ./src/main_gnn.py --config=$config_file --id=$commit_id --ts=$time_stamp --dir=$output_dir"board/" >> $std_file 2>&1 &

pid=$!

echo "Stdout dir:   $std_file"
echo "Start time:   `date -d @$time_stamp  '+%Y-%m-%d %H:%M:%S'`"
echo "CUDA DEVICES: $CUDA_VISIBLE_DEVICES"
echo "pid:          $pid"
cat $config_file

tail -f $std_file
