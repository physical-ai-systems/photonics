#!/bin/bash -l
#SBATCH --job-name=0.01_photonics
#SBATCH --nodes=1

# For h200
#SBATCH --gres=gpu:h200:4
#SBATCH --time=24:00:00
#SBATCH --output=log.log
#SBATCH --error=log.log


NUM_GPUS=4
# MIXED_PRECISION="bf16"
# MIXED_PRECISION="fp16"
MIXED_PRECISION="no"
GRADIENT_ACCUMULATION_STEPS=1
MAIN_PATH="/home/hpc/iwnt/iwnt153h"
REPO_DIR=$MAIN_PATH"/photonics"
BATCH_SIZE=1500
EPOCHS=200
LEARNING_RATE=1e-4
CLIP_MAX_NORM=1


cd $REPO_DIR
# Activate the conda environment
source ~/miniconda/bin/activate photonics


# Print configuration
echo "=================================="
echo "Distributed Training Configuration"
echo "=================================="
echo "Number of GPUs:              $NUM_GPUS"
echo "Mixed Precision:             $MIXED_PRECISION"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
# echo "Config:                      $CONFIG"
# echo "Experiment:                  $EXPERIMENT"
echo "Batch Size (per GPU):        $BATCH_SIZE"
echo "Test Batch Size:             $BATCH_SIZE"
echo "Effective Batch Size:        $((NUM_GPUS * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "Epochs:                      $EPOCHS"
echo "Learning Rate:               $LEARNING_RATE"
# echo "Aux Learning Rate:           $AUX_LEARNING_RATE"
echo "Clip Max Norm:               $CLIP_MAX_NORM"
# echo "Main Path:                   $MAIN_PATH"
# echo "Dataset:                     ${DATASET:-not specified}"
# echo "Dataset CSV:                 ${DATASET_CSV:-not specified}"
echo "=================================="
echo ""

# # Check if accelerate is installed
# if ! command -v accelerate &> /dev/null; then
#     echo "Error: accelerate is not installed"
#     echo "Please install it: pip install accelerate"
#     exit 1
# fi

# # Launch training
# echo "Launching distributed training..."
# echo ""


# Accelerate modes: eager, aot_eager, inductor, aot_ts_nvfuser, nvprims_nvfuser, cudagraphs, ofi, fx2trt, onnxrt, tensorrt, aot_torchxla_trace_once, torhchxla_trace_once, ipex, tvm

# Build the command with optional parameters
TRAIN_CMD="accelerate launch \
  --num_processes=$NUM_GPUS \
  --num_machines=1 \
  --dynamo_backend="no" \
  --mixed_precision=$MIXED_PRECISION \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  Scripts/train.py \
  --batch_size $BATCH_SIZE \
  --test_batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --learning_rate $LEARNING_RATE \
  --clip_max_norm $CLIP_MAX_NORM \
  --mixed_precision $MIXED_PRECISION"

  # --config $CONFIG \
  # --exp $EXPERIMENT \
  # --aux_learning_rate $AUX_LEARNING_RATE \
  # --main_path $MAIN_PATH \
  # --dataset $DATASET \
  # --dataset_csv $DATASET_CSV \

# Execute the command
eval $TRAIN_CMD

echo ""
echo "Training completed!"
