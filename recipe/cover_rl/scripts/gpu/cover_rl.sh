#!/bin/bash
n_gpus_per_node=4
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HYDRA_FULL_ERROR=1

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)
TRAIN_TASK="MATH-7500"
VAL_TASK=(MATH-TTT AMC-TTT AIME-TTT GPQA-TTT)
BACKBONE="your backbone"
BACKBONE_PATH="path to your backbone"

NUM_TURNS=2
INITIAL_ROLLOUT_N=32
INITIAL_ROLLOUT_SAMPLE_N=16

VERIFIER_ROLLOUT_MAX_N=8
REGENERATOR_ROLLOUT_N=6

USE_ANCHOR_FOR_VERIFIER=true
DOUBLE_CHECK="all"
CHOOSE_WRONG_NODE_STRATEGY="least_ratio"

CLIP_RATIO_HIGH=0.28
MULTI_TURN_EVAL=true
EVAL_MAX_TURNS=2
ENABLE_THINKING=true

CONFIG_INFO="uafv-${USE_ANCHOR_FOR_VERIFIER}-dc-${DOUBLE_CHECK}-et-${ENABLE_THINKING}-cwns-${CHOOSE_WRONG_NODE_STRATEGY}"

WANDB_PROJECT="CoVerRL"
LOCAL_DIR="$(pwd)/recipe/cover_rl/data"
VAL_TASK_STR=$(IFS='-'; echo "${VAL_TASK[*]}")
EXP_NAME="${BACKBONE}-${CONFIG_INFO}-${DATE}-${TIME_TAG}"
OUTPUT_DIR="checkpoints/${WANDB_PROJECT}/${EXP_NAME}"

VAL_FILES=""
for task in "${VAL_TASK[@]}"; do
  if [ -n "$VAL_FILES" ]; then
    VAL_FILES="${VAL_FILES},"
  fi
  VAL_FILES="${VAL_FILES}\"$LOCAL_DIR/$task/test.parquet\""
done

python -m recipe.cover_rl.main_cover_rl \
--config-name='ppo_trainer_cover_rl.yaml'\
  data.train_files=["$LOCAL_DIR/$TRAIN_TASK/math7500_train.parquet"] \
  data.val_files=[$VAL_FILES] \
  data.max_prompt_length=8192 \
  data.max_response_length=2048 \
  data.train_batch_size=64 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=16 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.max_model_len=16384 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=$BACKBONE_PATH \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=16 \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="$(pwd)/recipe/cover_rl/utils/reward_score/ttrl_math/__init__.py" \
  custom_reward_function.name=reward_func \
  cover_rl.enable=True \
  cover_rl.num_turns=$NUM_TURNS \
  cover_rl.initial_rollout_n=$INITIAL_ROLLOUT_N \
  cover_rl.initial_rollout_sample_n=$INITIAL_ROLLOUT_SAMPLE_N \
  cover_rl.use_anchor_for_verifier=$USE_ANCHOR_FOR_VERIFIER \
  cover_rl.double_check=$DOUBLE_CHECK \
  cover_rl.verify_regenerate.verifier_rollout_max_n=$VERIFIER_ROLLOUT_MAX_N \
  cover_rl.verify_regenerate.regenerator_rollout_n=$REGENERATOR_ROLLOUT_N \
  cover_rl.verify_regenerate.choose_wrong_node_strategy=$CHOOSE_WRONG_NODE_STRATEGY \
  cover_rl.multi_turn_eval=$MULTI_TURN_EVAL \
  cover_rl.eval_max_turns=$EVAL_MAX_TURNS \
  cover_rl.enable_thinking=$ENABLE_THINKING \
  trainer.logger=['console'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$EXP_NAME \
  trainer.n_gpus_per_node=$n_gpus_per_node \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=-1 \
  trainer.resume_mode=auto \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.val_before_train=True \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=2 "$@" | tee "$(pwd)/${WANDB_PROJECT}_${EXP_NAME}.log"
