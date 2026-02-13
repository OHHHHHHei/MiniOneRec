accelerate launch --num_processes 4 ./rq/text2emb/amazon_text2emb.py \
    --dataset Toys_and_Games \
    --root ./data/Amazon18/Toys_and_Games \
    --plm_checkpoint ../Qwen3-Embedding-8B
