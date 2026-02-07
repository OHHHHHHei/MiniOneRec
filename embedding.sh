# 你的项目根目录
PROJECT_ROOT="/mnt/shanhai-ai/qiuchenhao/leejt/MiniOneRec-2"
# 你要处理的数据集列表
TARGET_DATASETS=("Beauty")
# Qwen 模型路径
PLM_MODEL="../Qwen3-Embedding-8B"
export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"
# ==========================================
cd ${PROJECT_ROOT}
echo "当前工作目录: $(pwd)"

for DATASET in "${TARGET_DATASETS[@]}"; do
    echo "========================================================"
    echo "正在处理数据集: ${DATASET}"
    echo "========================================================"
    # ------------------------------------------------------
    # Step 2: 生成语义 Embedding (多卡并行)
    # ------------------------------------------------------
    echo "[Step 2] Generating Embeddings..."
    accelerate launch --num_processes 7 rq/text2emb/amazon_text2emb.py \
        --dataset ${DATASET} \
        --root ./data/Amazon/${DATASET} \
        --plm_checkpoint ${PLM_MODEL} \
        --plm_name qwen \
        --batch_size 64 
done

echo "所有数据集全部处理完毕!"
