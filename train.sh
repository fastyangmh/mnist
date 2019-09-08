#!/bin/bash

#parameters
multi_mode=$1
data_path=$2

#train
python ./src/train.py $multi_mode $data_path
