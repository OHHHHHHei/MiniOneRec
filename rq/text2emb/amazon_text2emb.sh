accelerate launch --num_processes 8 ./rq/text2emb/amazon_text2emb.py \
    --dataset Sports \
    --root ./data/Amazon18/Sports \
    --plm_checkpoint ../Qwen3-Embedding-8B
