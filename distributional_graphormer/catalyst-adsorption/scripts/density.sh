#!/bin/bash

echo "============================== Git Commit Head =============================="
git rev-parse HEAD
echo "============================================================================="

WANDB_PROJECT=none
NUM_GPUS=$1
MAX_EPOCH=1
BATCH_SIZE=$2
NUM_DIFFUSION_TIMESTEPS=5000
SAMPLING=ddpm
DIFFUSION_BETA_END=2e-3
NO_DIFFUSION=false
KDE_TEMPERATURE=1.0
SAVE_DIR=$3
PBC_APPROACH=cutoff
TRAIN_NAME=train-100 # dummy training data for sampling
NUM_TRAIN_DATA=100
PBC_CUTOFF=6.0
DIFFUSION_NOISE_STD=1.0
VAL_SET_NAME=grid-30457 # dummy validation data for sampling
D_NODE=1
D_PROC=${NUM_GPUS}
DDP_BACKEND=legacy_ddp
FP16=true
FINETUNE_FROM_MODEL=checkpoints/checkpoint.pt
REMOVE_HEAD=false
USE_BONDS=true
CRITERION=flow_ode_calc_density
NUM_EPSILON_ESTIMATOR=1
N_KDE_SAMPLES=1
SEED=0
ARCH=base
LR=2e-4 # dummy learning rate
DENSITY_CALC_Z_OFFSET=$4

NUM_SAMPLES=${NUM_TRAIN_DATA}
MAX_UPDATE=1
WARMUP_UPDATES=$((${MAX_UPDATE} * 6 / 100))

[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=0.0.0.0

[ -z "${MASTER_PORT}" ] && MASTER_PORT=10086

[ -z "${RANK}" ] && RANK=0

[ -z "${FP16}" ] && FP16=true

[ -z "${FINETUNE_FROM_MODEL}" ] && FINETUNE_FROM_MODEL=none

[ -z "${REMOVE_HEAD}" ] && REMOVE_HEAD=false

[ -z "${USE_BONDS}" ] && USE_BONDS=false

[ -z "${DENSITY_CALC_Z_OFFSET}" ] && DENSITY_CALC_Z_OFFSET=0

echo "============================== PARAMS =============================="
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "MAX_EPOCH=${MAX_EPOCH}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "NUM_DIFFUSION_TIMESTEPS=${NUM_DIFFUSION_TIMESTEPS}"
echo "SAMPLING=${SAMPLING}"
echo "DIFFUSION_BETA_END=${DIFFUSION_BETA_END}"
echo "NUM_SAMPLES=${NUM_SAMPLES}"
echo "MAX_UPDATE=${MAX_UPDATE}"
echo "WARMUP_UPDATES=${WARMUP_UPDATES}"
echo "NO_DIFFUSION=${NO_DIFFUSION}"
echo "KDE_TEMPERATURE=${KDE_TEMPERATURE}"
echo "SAVE_DIR=${SAVE_DIR}"
echo "PBC_APPROACH=${PBC_APPROACH}"
echo "TRAIN_NAME=${TRAIN_NAME}"
echo "NUM_TRAIN_DATA=${NUM_TRAIN_DATA}"
echo "PBC_CUTOFF=${PBC_CUTOFF}"
echo "DIFFUSION_NOISE_STD=${DIFFUSION_NOISE_STD}"
echo "VAL_SET_NAME=${VAL_SET_NAME}"
echo "D_NODE=${D_NODE}"
echo "D_PROC=${D_PROC}"
echo "DDP_BACKEND=${DDP_BACKEND}"
echo "FP16=${FP16}"
echo "FINETUNE_FROM_MODEL=${FINETUNE_FROM_MODEL}"
echo "REMOVE_HEAD=${REMOVE_HEAD}"
echo "USE_BONDS=${USE_BONDS}"
echo "CRITERION=${CRITERION}"
echo "NUM_EPSILON_ESTIMATOR=${NUM_EPSILON_ESTIMATOR}"
echo "N_KDE_SAMPLES=${N_KDE_SAMPLES}"
echo "SEED=${SEED}"
echo "LR=${LR}"
echo "DENSITY_CALC_Z_OFFSET=${DENSITY_CALC_Z_OFFSET}"
echo "====================================================================="

echo -e "\n\n"
echo "==================================MP==========================================="
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
echo "OMPI_COMM_WORLD_RANK: ${OMPI_COMM_WORLD_RANK}"
echo "OMPI_COMM_WORLD_SIZE: ${OMPI_COMM_WORLD_SIZE}"

ddp_options="--nnodes=$D_NODE --node_rank=$RANK --master_addr=$MASTER_ADDR"
echo "ddp_options: ${ddp_options}"
echo "==============================================================================="



if [ ${NO_DIFFUSION} == "true" ]; then
    NO_DIFFUSION="--no-diffusion"
else
    NO_DIFFUSION=""
fi

if [ ${FP16} == "true" ]; then
    FP16="--fp16"
else
    FP16=""
fi

if [ ${FINETUNE_FROM_MODEL} == "none" ]; then
    FINETUNE_FROM_MODEL=""
else
    FINETUNE_FROM_MODEL="--finetune-from-model ${FINETUNE_FROM_MODEL}"
fi

if [ ${REMOVE_HEAD} == "true" ]; then
    REMOVE_HEAD="--remove-head"
else
    REMOVE_HEAD=""
fi

if [ ${USE_BONDS} == "true" ]; then
    USE_BONDS="--use-bonds"
else
    USE_BONDS=""
fi

if [ ${CRITERION} == "flow_ode_calc_density" ]; then
    DENSITY_CALC_Z_OFFSET="--density-calc-z-offset ${DENSITY_CALC_Z_OFFSET}"
else
    DENSITY_CALC_Z_OFFSET=""
fi


export NCCL_ASYNC_ERROR_HADNLING=1
export OMP_NUM_THREADS=1

python -m torch.distributed.launch --nproc_per_node=$D_PROC --master_port=$MASTER_PORT $ddp_options \
    $(which fairseq-train) \
    --user-dir ./graphormer --ddp-backend ${DDP_BACKEND} --task graph_diffusion \
    --data-path ./dataset/ \
    --arch graphormer_diff_${ARCH} \
    --num-workers 16 \
    --train-subset ${TRAIN_NAME} \
    --valid-subset ${VAL_SET_NAME} \
    --batch-size ${BATCH_SIZE} \
    --validate-interval 1 \
    --max-update ${MAX_UPDATE} \
    --max-epoch ${MAX_EPOCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr ${LR} \
    --lr-scheduler polynomial_decay \
    --num-diffusion-timesteps ${NUM_DIFFUSION_TIMESTEPS} \
    --diffusion-beta-schedule sigmoid \
    --diffusion-sampling ${SAMPLING} \
    --ddim-steps 50 \
    --diffusion-beta-end ${DIFFUSION_BETA_END} \
    --warmup-updates ${WARMUP_UPDATES} \
    --total-num-update ${MAX_UPDATE} \
    --keep-best-checkpoints 5 \
    --keep-last-epochs 5 \
    --save-dir ${SAVE_DIR} \
    --best-checkpoint-metric kde_kl \
    --criterion ${CRITERION} \
    --kde-temperature ${KDE_TEMPERATURE} \
    --pbc-cutoff ${PBC_CUTOFF} \
    --pbc-approach ${PBC_APPROACH} \
    --diffusion-noise-std ${DIFFUSION_NOISE_STD} ${NO_DIFFUSION} ${FP16} ${REMOVE_HEAD} ${FINETUNE_FROM_MODEL} ${USE_BONDS} ${DENSITY_CALC_Z_OFFSET} \
    --batch-size-valid ${BATCH_SIZE} \
    --num-epsilon-estimator ${NUM_EPSILON_ESTIMATOR} \
    --n-kde-samples ${N_KDE_SAMPLES} \
    --result-save-dir ${SAVE_DIR} \
    --seed ${SEED}
