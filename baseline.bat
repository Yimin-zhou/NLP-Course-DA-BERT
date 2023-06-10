@echo off
set train_data=data\baseline\train_100.csv
set valid_data=data\baseline\train_1000.csv
set test_data=data\baseline\test_2000.csv
set model=bert-base-uncased
set max_len=512
set epochs=5
set train_batch=32
set test_batch=64
set output=output\baseline

python baseline.py ^
    --train_data %train_data% ^
    --valid_data %valid_data% ^
    --test_data %test_data% ^
    --model %model% ^
    --max_len %max_len% ^
    --epochs %epochs% ^
    --train_batch %train_batch% ^
    --test_batch %test_batch% ^
    --output %output%




