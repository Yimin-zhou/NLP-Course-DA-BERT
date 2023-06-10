
python baseline.py \
    --train_data data/baseline/train_100.csv \
    --valid_data data/baseline/train_1000.csv \
    --test_data data/baseline/test_2000.csv \
    --model bert-base-uncased \
    --max_len 512 \
    --epochs 5 \
    --train_batch 64 \
    --test_batch 64 \
    --output output/baseline
