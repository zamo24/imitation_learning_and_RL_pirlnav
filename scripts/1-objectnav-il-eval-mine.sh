export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

config="configs/experiments/il_objectnav.yaml"

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1"
TENSORBOARD_DIR="tb/eval/"
EVAL_CKPT_PATH_DIR=data/new_checkpoints/objectnav_il_new

mkdir -p $TENSORBOARD_DIR
set -x

echo "In ObjectNav IL eval"
torchrun --nproc_per_node=1 -m run \
--exp-config $config \
--run-type eval \
TENSORBOARD_DIR $TENSORBOARD_DIR \
EVAL_CKPT_PATH_DIR $EVAL_CKPT_PATH_DIR \
NUM_ENVIRONMENTS 16 \
RL.DDPPO.force_distributed True \
TASK_CONFIG.DATASET.SPLIT "val" \
EVAL.USE_CKPT_CONFIG True \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TEST_EPISODE_COUNT 500
