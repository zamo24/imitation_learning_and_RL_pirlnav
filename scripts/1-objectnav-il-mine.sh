
#!/bin/bash

NUM_ENVIRONMENTS=8

# Set environment variables to reduce logging noise
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# Navigate to the project directory
# cd $PROJECT_ROOT

# --- Experiment Configuration ---
config="configs/experiments/il_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d_hd/"
TENSORBOARD_DIR="tb/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/"
INFLECTION_COEF=3.234951275740812

# Create output directories if they don't exist
mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR

set -x

echo "Starting ObjectNav IL run on a local machine..."

torchrun --nproc_per_node=8 -m run \
--exp-config $config \
--run-type train \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
NUM_UPDATES 100000 \
NUM_ENVIRONMENTS $NUM_ENVIRONMENTS \
RL.DDPPO.force_distributed True \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF \
NUM_CHECKPOINTS 500
