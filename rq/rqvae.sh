python rq/rqvae.py \
      --data_path ./data/Amazon18/Toys_and_Games/Toys_and_Games.emb-qwen-td.npy \
      --ckpt_dir ./output/Toys_and_Games \
      --lr 1e-3 \
      --epochs 10000 \
      --batch_size 20480
