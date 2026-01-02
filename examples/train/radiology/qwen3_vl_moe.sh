# If you don't want to train the router, set:
# `--target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`

# Note: If you need to use DeepSpeed ZeRO-2/ZeRO-3 but encounter hangs
# try using transformers==4.51.3

NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model /home/user/checkpoints/qwen3-vl-30b-a3b-instruct-merged \
    --template qwen3_vl_custom \
    --train_type full \
    --dataset examples/custom/dummy_data/ \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --freeze_vit true \
    --freeze_llm true \
    --freeze_aligner false \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --router_aux_loss_coef 1e-3 \
    --gradient_accumulation_steps 1 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.0 \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
    --model_author swift \
    --model_name swift-robot
