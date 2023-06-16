@echo off
set train_size=1000
set train_data=data\processed\baseline\train_%train_size%.csv
set valid_data=data\processed\baseline\test_2000.csv
set test_data=data\processed\baseline\test_2000.csv
set model=bert-base-uncased
set max_len=512
set epochs=5
set train_batch=4
set test_batch=4
set output=output\latent\train_%train_size%_4_20

python extrapolation.py ^
    --train_data %train_data% ^
    --valid_data %valid_data% ^
    --test_data %test_data% ^
    --model %model% ^
    --max_len %max_len% ^
    --epochs %epochs% ^
    --train_batch %train_batch% ^
    --test_batch %test_batch% ^
    --output %output%




