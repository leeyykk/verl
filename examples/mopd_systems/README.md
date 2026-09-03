# MOPD Systems Presets

This directory contains branch-local launchers for system characterization of
multi-teacher on-policy distillation on one 8 GPU node.

The default objective follows the original MOPD top-k support direction:

```text
student rollout -> routed teacher top-k -> student scores teacher top-k ids
```

This differs from the Open-MOPD implementation we inspected earlier:

```text
student rollout -> student top-k -> routed teacher scores student top-k ids
```

The implementation uses verl's existing on-policy distillation path:

- teacher routing key: `distillation.teacher_key`
- teacher outputs: `teacher_ids`, `teacher_logprobs`
- top-k loss: `distillation.distillation_loss.loss_mode=forward_kl_topk`

## 8 GPU Modes

`run_qwen3_8b_mopd_paper_topk_8gpu.sh` supports modes whose requested Ray
resource pools sum to 8 GPUs:

```text
SYSTEM_MODE=2-2-2-2
  2 GPUs: student actor/rollout/training pool
  2 GPUs: math teacher
  2 GPUs: code teacher
  2 GPUs: instruction-following teacher

SYSTEM_MODE=4s4t
  4 GPUs: student actor/rollout/training pool
  2 GPUs: math teacher
  1 GPU : code teacher
  1 GPU : instruction-following teacher
```

SYSTEM_MODE=colocated8
  8 GPUs: student actor/rollout/training pool
  8 GPUs: math teacher, colocated on the same physical pool
  8 GPUs: code teacher, colocated on the same physical pool
  8 GPUs: instruction-following teacher, colocated on the same physical pool

`colocated8` is Open-MOPD-style time-sharing: it requests one physical 8 GPU
Ray pool, maps teacher servers to that same pool, and lets each teacher reuse
the full pool instead of splitting the pool across teachers. This is useful for
systems characterization on a single node, but VRAM pressure is real because the
student rollout/training engine and teacher inference engines all coexist.

## Example

```bash
cd /NHNHOME/home/verl
SYSTEM_MODE=2-2-2-2 \
STUDENT_MODEL=/NHNHOME/home/huggingface_models/Qwen3-8B \
MATH_TEACHER_MODEL=/NHNHOME/home/huggingface_models/Qwen2.5-32B-Instruct \
CODE_TEACHER_MODEL=/NHNHOME/home/huggingface_models/Qwen2.5-32B-Instruct \
IF_TEACHER_MODEL=/NHNHOME/home/huggingface_models/Qwen2.5-32B-Instruct \
TRAIN_FILE=/NHNHOME/home/Open-MOPD/data/routing_probe/train_24_balanced.parquet \
VAL_FILE=/NHNHOME/home/Open-MOPD/data/routing_probe/val.parquet \
TOTAL_EPOCHS=1 \
TRAIN_BATCH_SIZE=24 \
bash examples/mopd_systems/run_qwen3_8b_mopd_paper_topk_8gpu.sh
```

## Timeline Output

The launcher writes per-event JSONL to:

```text
${RUN_DIR}/timeline.jsonl
```

Set `RUN_DIR=/path/to/run` or `TIMELINE_JSONL=/path/to/timeline.jsonl` to
choose the output path. The events currently include per-sample student rollout
start/end, routed teacher top-k start/end, and student training start/end.

## Colocated Example

```bash
cd /NHNHOME/home/verl
SYSTEM_MODE=colocated8 \
STUDENT_MODEL=/NHNHOME/home/huggingface_models/Qwen3-8B \
MATH_TEACHER_MODEL=/NHNHOME/home/huggingface_models/Qwen2.5-32B-Instruct \
CODE_TEACHER_MODEL=/NHNHOME/home/huggingface_models/Qwen2.5-32B-Instruct \
IF_TEACHER_MODEL=/NHNHOME/home/huggingface_models/Qwen2.5-32B-Instruct \
TRAIN_FILE=/NHNHOME/home/Open-MOPD/data/routing_probe/train_24_balanced.parquet \
VAL_FILE=/NHNHOME/home/Open-MOPD/data/routing_probe/val.parquet \
TOTAL_EPOCHS=1 \
TOTAL_TRAINING_STEPS=4 \
TRAIN_BATCH_SIZE=24 \
bash examples/mopd_systems/run_qwen3_8b_mopd_paper_topk_8gpu.sh
```
