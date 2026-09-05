#!/usr/bin/env bash

MODEL=TODO
CONFIG=TODO

export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20

export CUDA_VISIBLE_DEVICES=TODO
GPUS=TODO

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29502}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

TIME=$(date "+%Y%m%d-%H%M%S")

JOB_NAME=${CONFIG}[${TIME}]
CONFIG_FILE="configs/${MODEL}/${CONFIG}.py" 
WORK_DIR="work_dirs/${JOB_NAME}"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    --config ${CONFIG_FILE} \
    --work-dir=${WORK_DIR} \
    --seed 0 \
    --launcher pytorch ${@:3}\
    --no-validate
