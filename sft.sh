export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
# Office_Products, Industrial_and_Scientific Sports Toys_and_Games
for category in "Industrial_and_Scientific" ; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    
    torchrun --nproc_per_node 4 \
            sft.py \
            --base_model ../Qwen3-1.7B \
            --batch_size 1024 \
            --micro_batch_size 4 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir output/sft_${category}_1.7B_noalign_2 \
            --wandb_project MiniOneRec \
            --wandb_run_name train_${category}_A100_noalign \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/${category}.index.json \
            --item_meta_path ./data/Amazon/index/${category}.item.json \
            --freeze_LLM False
done
