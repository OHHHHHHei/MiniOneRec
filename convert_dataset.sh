#!/bin/bash


PYTHON_SCRIPT="convert_dataset.py"

INPUT_DIR="data/Amazon18/Toys_and_Games"

OUTPUT_DIR="data/Amazon"

DATASET_NAME="Toys_and_Games"

# ===========================================

echo "Start converting $DATASET_NAME ..."

python $PYTHON_SCRIPT \
    --dataset_name $DATASET_NAME \
    --data_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --category $DATASET_NAME \
    --seed 42

echo "Finished!"
