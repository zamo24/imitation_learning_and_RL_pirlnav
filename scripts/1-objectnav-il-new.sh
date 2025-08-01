#!/bin/bash

# A short description of the user's request.
# The user wants to run an Imitation Learning (IL) experiment on their local machine,
# which has one GPU and does not use the Slurm workload manager.

# --- USER-CONFIGURABLE-SECTION ---
# IMPORTANT: Please update these paths to match your local setup.

# 1. Path to your conda installation's profile script.
#    This is often in your miniconda3 or anaconda3 directory.
CONDA_PROFILE_PATH="/srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh" # EDIT THIS

# 2. The absolute path to your project's root directory.
PROJECT_ROOT="/srv/flash1/rramrakhya6/spring_2022/pirlnav" # EDIT THIS

# 3. The name of the conda environment that has the required dependencies.
CONDA_ENV_NAME="pirlnav"

# 4. Which GPU to use (0 for the first GPU, 1 for the second, etc.)
GPU_ID=0

# 5. Number of parallel environments. The original script used 16 for a powerful server.
#    For a local PC, a lower number is recommended to avoid running out of memory.
#    Adjust this based on your GPU's VRAM and system RAM. Start low.
NUM_ENVIRONMENTS=8
# --- END-USER-CONFIGURABLE-SECTION ---


# Activate the conda environment
# source $CONDA_PROFILE_PATH
# conda deactivate
# conda activate $CONDA_ENV_NAME

# Set environment variables to reduce logging noise
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# Navigate to the project directory
# cd $PROJECT_ROOT

# --- Experiment Configuration ---
# These are taken from your original script.
# You can modify them here if needed.
config="configs/experiments/il_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d_hd/"
TENSORBOARD_DIR="tb/new"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il_new/"
INFLECTION_COEF=3.234951275740812

# Create output directories if they don't exist
mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR

# The `set -x` command will print each command before it is executed, which is useful for debugging.
set -x

echo "Starting ObjectNav IL run on a local machine..."

# Execute the training command
# - `CUDA_VISIBLE_DEVICES=$GPU_ID` tells the script which specific GPU to use.
# - We removed `srun` as it's part of Slurm.
# - `RL.DDPPO.force_distributed` is set to `False` for single-GPU training.
torchrun --nproc_per_node=8 -m run \
--exp-config $config \
--run-type train \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
NUM_UPDATES 100000 \
NUM_CHECKPOINTS 1000 \
NUM_ENVIRONMENTS $NUM_ENVIRONMENTS \
RL.DDPPO.force_distributed True \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF
