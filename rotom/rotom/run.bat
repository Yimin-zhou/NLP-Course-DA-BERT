set CUDA_VISIBLE_DEVICES=0
python train_any.py --task textcls_AMAZON2 --size 300 --logdir results_em/ --finetuning --batch_size 32 --lr 3e-5 --n_epochs 20 --max_len 128 --fp16 --lm roberta --da auto_filter_weight --balance --run_id 0
