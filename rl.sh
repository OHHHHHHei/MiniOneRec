#!/bin/bash

export NCCL_IB_DISABLE=1

# 在这里列出所有你想要运行的类别
for category in "Toys_and_Games" ; do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    echo "正在开始训练类别: ${category}"

    HF_ENDPOINT=https://hf-mirror.com accelerate launch \
        --config_file ./config/zero2_opt.yaml \
        --num_processes 4 --main_process_port 29503 \
        rl.py \
        --model_path ./output/sft_${category}_1.7B_align/final_checkpoint \
        --train_batch_size 64 \
        --eval_batch_size 128 \
        --num_train_epochs 2 \
        --gradient_accumulation_steps 2 \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --info_file ${info_file} \
        --category ${category} \
        --sample_train False \
        --eval_step 0.0999 \
        --reward_type ranking \
        --num_generations 8 \
        --mask_all_zero False \
        --dynamic_sampling False \
        --sync_ref_model True \
        --beam_search True \
        --test_during_training False \
        --temperature 1.0 \
        --learning_rate 1e-5 \
        --add_gt False \
        --beta 1e-3 \
        --dapo False \
        --output_dir ./output/RL_${category}_1.7B_align \
        --wandb_project MiniOneRec \
        --wandb_run_name RL_${category}_1.7B_align \
        --resume_from_checkpoint ./output/RL_Toys_and_Games_1.7B_align/checkpoint-1800 \
        --sid_index_path ./data/Amazon18/Toys_and_Games/Toys_and_Games.index.json \
        --item_meta_path ./data/Amazon18/Toys_and_Games/Toys_and_Games.item.json 
done