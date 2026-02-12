# sh examples/custom/sft-ct.sh
#
# Uses --streaming so that CT preprocessing (R2 download + DICOM
# extraction) happens lazily per-row inside the *template* encode(),
# driven by the standard EncodePreprocessor — one map, no chaining.
export WANDB_PROJECT=smb-vision

nproc_per_node=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --external_plugins examples/custom/ct_dataset.py \
    --model google/medgemma-1.5-4b-it \
    --use_hf true \
    --template ct_gemma3_vision \
    --dataset report_generation \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --attn_impl flash_attn \
    --learning_rate 1e-5 \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 64  \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner false \
    --freeze_llm false \
    --gradient_checkpointing false \
    --vit_gradient_checkpointing false \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_strategy 'no' \
    --eval_steps 100 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --max_length 11520 \
    --output_dir output \
    --dataset_num_proc 8 \
    --deepspeed zero2 \
    --report_to wandb \
    --run_name medgemma-1.5-4b-it-sft

# 2 * 23GiB; 2.3s/it
# PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
# NPROC_PER_NODE=8 \
# MAX_PIXELS=1003520 \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# megatron sft \
#     --external_plugins examples/custom/ct_dataset.py \
#     --template_type ct_gemma3_vision \
#     --model google/medgemma-1.5-4b-it \
#     --load_safetensors true \
#     --save_safetensors true \
#     --use_hf true \
#     --merge_lora false \
#     --dataset report_generation \
#     --load_from_cache_file true \
#     --tuner_type lora \
#     --lora_rank 16 \
#     --lora_alpha 64 \
#     --target_modules all-linear \
#     --tensor_model_parallel_size 1 \
#     --sequence_parallel true \
#     --freeze_llm false \
#     --freeze_vit true \
#     --freeze_aligner false \
#     --packing true \
#     --split_dataset_ratio 0.01 \
#     --micro_batch_size 1 \
#     --global_batch_size 16 \
#     --recompute_granularity full \
#     --recompute_method uniform \
#     --recompute_num_layers 1 \
#     --finetune true \
#     --cross_entropy_loss_fusion true \
#     --lr 1e-5 \
#     --lr_warmup_fraction 0.05 \
#     --min_lr 1e-6 \
#     --max_epochs 1 \
#     --save megatron_output/medgemma-1.5-4b-it \
#     --save_interval 200 \
#     --vit_gradient_checkpointing true \
#     --max_length 25600 \
#     --num_workers 4 \
#     --no_save_optim true \
#     --no_save_rng true \
#     --dataset_num_proc 8
