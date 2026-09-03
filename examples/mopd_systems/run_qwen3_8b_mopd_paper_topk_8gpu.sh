#!/usr/bin/env bash
# Paper-style MOPD top-k support on one 8 GPU node.
#
# Objective direction:
#   student rollout -> routed teacher top-k -> student scores teacher top-k ids
#
# This uses verl's native on-policy distillation teacher loop rather than the
# Open-MOPD student-top-k reward path.

set -xeuo pipefail

SYSTEM_MODE=${SYSTEM_MODE:-2-2-2-2}

STUDENT_MODEL=${STUDENT_MODEL:-/NHNHOME/home/huggingface_models/Qwen3-8B}
MATH_TEACHER_MODEL=${MATH_TEACHER_MODEL:-/NHNHOME/home/huggingface_models/Qwen2.5-32B-Instruct}
CODE_TEACHER_MODEL=${CODE_TEACHER_MODEL:-/NHNHOME/home/huggingface_models/Qwen2.5-32B-Instruct}
IF_TEACHER_MODEL=${IF_TEACHER_MODEL:-/NHNHOME/home/huggingface_models/Qwen2.5-32B-Instruct}

TRAIN_FILE=${TRAIN_FILE:-/NHNHOME/home/Open-MOPD/data/routing_probe/train_24_balanced.parquet}
VAL_FILE=${VAL_FILE:-/NHNHOME/home/Open-MOPD/data/routing_probe/val.parquet}

NNODES=${NNODES:-1}
TEACHER_NNODES=${TEACHER_NNODES:-1}

case "$SYSTEM_MODE" in
  2-2-2-2)
    STUDENT_GPUS=${STUDENT_GPUS:-2}
    ROLLOUT_TP=${ROLLOUT_TP:-2}
    MATH_TEACHER_TP=${MATH_TEACHER_TP:-2}
    CODE_TEACHER_TP=${CODE_TEACHER_TP:-2}
    IF_TEACHER_TP=${IF_TEACHER_TP:-2}
    ;;
  4s4t|4-4)
    STUDENT_GPUS=${STUDENT_GPUS:-4}
    ROLLOUT_TP=${ROLLOUT_TP:-2}
    MATH_TEACHER_TP=${MATH_TEACHER_TP:-2}
    CODE_TEACHER_TP=${CODE_TEACHER_TP:-1}
    IF_TEACHER_TP=${IF_TEACHER_TP:-1}
    ;;
  colocated8|8g-colocated)
    STUDENT_GPUS=${STUDENT_GPUS:-8}
    ROLLOUT_TP=${ROLLOUT_TP:-8}
    MATH_TEACHER_TP=${MATH_TEACHER_TP:-8}
    CODE_TEACHER_TP=${CODE_TEACHER_TP:-8}
    IF_TEACHER_TP=${IF_TEACHER_TP:-8}
    COLOCATE_TEACHERS=True
    REUSE_FULL_POOL_PER_TEACHER=True
    MOPD_MAX_COLOCATE_COUNT=${MOPD_MAX_COLOCATE_COUNT:-6}
    ;;
  *)
    echo "Unknown SYSTEM_MODE=$SYSTEM_MODE. Use 2-2-2-2, 4s4t, or colocated8." >&2
    exit 2
    ;;
esac

COLOCATE_TEACHERS=${COLOCATE_TEACHERS:-False}
REUSE_FULL_POOL_PER_TEACHER=${REUSE_FULL_POOL_PER_TEACHER:-False}
MOPD_MAX_COLOCATE_COUNT=${MOPD_MAX_COLOCATE_COUNT:-3}
if [ "$COLOCATE_TEACHERS" = True ]; then
  TEACHER_WORLD_SIZE=${STUDENT_GPUS}
  if [ "$STUDENT_GPUS" -ne 8 ]; then
    echo "SYSTEM_MODE=$SYSTEM_MODE is for one 8 GPU node, but STUDENT_GPUS=$STUDENT_GPUS." >&2
    exit 2
  fi
else
  TEACHER_WORLD_SIZE=$((MATH_TEACHER_TP + CODE_TEACHER_TP + IF_TEACHER_TP))
  TOTAL_GPUS=$((STUDENT_GPUS + TEACHER_WORLD_SIZE))
  if [ "$TOTAL_GPUS" -ne 8 ]; then
    echo "This launcher is for one 8 GPU node, but STUDENT_GPUS + teacher TP = $TOTAL_GPUS." >&2
    exit 2
  fi
fi

DISTILLATION_TOPK=${DISTILLATION_TOPK:-64}
DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-forward_kl_topk}
USE_TASK_REWARDS=${USE_TASK_REWARDS:-False}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-False}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-24}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-24}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-256}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.35}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.45}
ACTOR_LR=${ACTOR_LR:-1e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-4}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:--1}

PROJECT_NAME=${PROJECT_NAME:-verl_mopd_systems}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-paper_topk_${SYSTEM_MODE}_8g}

RUN_ROOT=${RUN_ROOT:-/NHNHOME/home/verl/runs/mopd_systems}
RUN_DIR=${RUN_DIR:-${RUN_ROOT}/${EXPERIMENT_NAME}}
TIMELINE_JSONL=${TIMELINE_JSONL:-${RUN_DIR}/timeline.jsonl}
mkdir -p "$RUN_DIR"
export VERL_MOPD_TIMELINE_JSONL="$TIMELINE_JSONL"

MAX_NUM_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))

DATA=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  data.train_files="['$TRAIN_FILE']"
  data.val_files="['$VAL_FILE']"
  data.train_batch_size=${TRAIN_BATCH_SIZE}
  data.max_prompt_length=${MAX_PROMPT_LENGTH}
  data.max_response_length=${MAX_RESPONSE_LENGTH}
  data.filter_overlong_prompts=True
  data.truncation='error'
  data.shuffle=False
)

MODEL=(
  actor_rollout_ref.model.path="$STUDENT_MODEL"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
  actor_rollout_ref.actor.use_torch_compile=False
  actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
  actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
  actor_rollout_ref.rollout.n=1
  actor_rollout_ref.rollout.max_model_len=${MAX_NUM_TOKENS}
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
)

TRAINER=(
  trainer.balance_batch=True
  trainer.logger='["console"]'
  trainer.project_name=${PROJECT_NAME}
  trainer.experiment_name=${EXPERIMENT_NAME}
  trainer.n_gpus_per_node=${STUDENT_GPUS}
  trainer.nnodes=${NNODES}
  trainer.val_before_train=False
  trainer.save_freq=${SAVE_FREQ}
  trainer.test_freq=${TEST_FREQ}
  trainer.total_epochs=${TOTAL_EPOCHS}
  trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
  +mopd_systems.colocate_teachers_with_actor_rollout=${COLOCATE_TEACHERS}
  +mopd_systems.reuse_full_pool_per_teacher=${REUSE_FULL_POOL_PER_TEACHER}
  +mopd_systems.max_colocate_count=${MOPD_MAX_COLOCATE_COUNT}
)

DISTILL=(
  distillation.enabled=True
  distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
  distillation.nnodes=${TEACHER_NNODES}
  distillation.teacher_key=domain
  +distillation.teacher_models.math.key=math
  +distillation.teacher_models.math.model_path="$MATH_TEACHER_MODEL"
  +distillation.teacher_models.math.num_replicas=1
  +distillation.teacher_models.math.inference.name=vllm
  +distillation.teacher_models.math.inference.tensor_model_parallel_size=${MATH_TEACHER_TP}
  +distillation.teacher_models.math.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}
  +distillation.teacher_models.math.inference.max_model_len=${MAX_NUM_TOKENS}
  +distillation.teacher_models.code.key=code
  +distillation.teacher_models.code.model_path="$CODE_TEACHER_MODEL"
  +distillation.teacher_models.code.num_replicas=1
  +distillation.teacher_models.code.inference.name=vllm
  +distillation.teacher_models.code.inference.tensor_model_parallel_size=${CODE_TEACHER_TP}
  +distillation.teacher_models.code.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}
  +distillation.teacher_models.code.inference.max_model_len=${MAX_NUM_TOKENS}
  +distillation.teacher_models.if.key=if
  +distillation.teacher_models.if.model_path="$IF_TEACHER_MODEL"
  +distillation.teacher_models.if.num_replicas=1
  +distillation.teacher_models.if.inference.name=vllm
  +distillation.teacher_models.if.inference.tensor_model_parallel_size=${IF_TEACHER_TP}
  +distillation.teacher_models.if.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}
  +distillation.teacher_models.if.inference.max_model_len=${MAX_NUM_TOKENS}
  distillation.distillation_loss.loss_mode=${DISTILLATION_LOSS_MODE}
  distillation.distillation_loss.topk=${DISTILLATION_TOPK}
  distillation.distillation_loss.use_task_rewards=${USE_TASK_REWARDS}
  distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT}
  distillation.distillation_loss.loss_max_clamp=10.0
  distillation.distillation_loss.log_prob_min_clamp=-10.0
)

LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
  LAUNCH=(uv run --frozen --all-packages --extra vllm --extra fsdp python3)
  RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp")
fi

"${LAUNCH[@]}" -m verl.trainer.main_ppo \
  "${DATA[@]}" \
  "${MODEL[@]}" \
  "${ACTOR[@]}" \
  "${ROLLOUT[@]}" \
  "${TRAINER[@]}" \
  "${DISTILL[@]}" \
  "${RAY[@]}" \
  "$@"
