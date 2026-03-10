#!/bin/bash
set -euo pipefail

# Simple launcher for FedPuReL on CIFAR-100-LT.
# Usage: ./scripts/cifar100.sh [peft_type] [config]
#   peft_type: prompt | lora | adapter (default: prompt)
#   config: trainer config under configs/trainers/FedPuReL (default: vit_b16)

PEFT_TYPE=${1:-prompt}
TRAINER_CFG=${2:-vit_b16}

DATA_ROOT=${DATA_ROOT:-DATA}
OUTPUT_ROOT=${OUTPUT_ROOT:-output/FedPuReL/cifar100}
DATASET=cifar100_LT
NUM_CLASSES=100
PARTITION=noniid-labeldir100_LT

NUM_USERS=${NUM_USERS:-20}
FRAC=${FRAC:-0.4}
ROUNDS=${ROUNDS:-100}
LR=${LR:-0.001}
SEED=${SEED:-1}
TRAIN_BATCH=${TRAIN_BATCH:-32}
TEST_BATCH=${TEST_BATCH:-128}
CSC=${CSC:-False}
CTX_INIT=${CTX_INIT:-False}
N_CTX=${N_CTX:-4}
N_GENERAL=${N_GENERAL:-0}
LORA_RANK=${LORA_RANK:-8}
FUSION_LOSS_ALPHA=${FUSION_LOSS_ALPHA:-0.99}
IMB_FACTOR=${IMB_FACTOR:-0.01}
BETA=${BETA:-1.0}

RUN_NAME="${PEFT_TYPE}_${TRAINER_CFG}_seed${SEED}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

CUDA_DEVICE=${CUDA_VISIBLE_DEVICES:-0}
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} \
python federated_main.py \
  --root "${DATA_ROOT}" \
  --model fedavg \
  --trainer FedPuReL \
  --peft_type "${PEFT_TYPE}" \
  --dataset "${DATASET}" \
  --dataset-config-file "configs/datasets/${DATASET}.yaml" \
  --config-file "configs/trainers/FedPuReL/${TRAINER_CFG}.yaml" \
  --output-dir "${OUTPUT_DIR}" \
  --seed ${SEED} \
  --num_users ${NUM_USERS} \
  --frac ${FRAC} \
  --round ${ROUNDS} \
  --lr ${LR} \
  --train_batch_size ${TRAIN_BATCH} \
  --test_batch_size ${TEST_BATCH} \
  --partition ${PARTITION} \
  --beta ${BETA} \
  --imb_factor ${IMB_FACTOR} \
  --num_classes ${NUM_CLASSES} \
  --n_ctx ${N_CTX} \
  --n_general ${N_GENERAL} \
  --csc ${CSC} \
  --ctx_init ${CTX_INIT} \
  --lora_rank ${LORA_RANK} \
  --fusion_loss_alpha ${FUSION_LOSS_ALPHA}
