

python main.py \
    --cfg='config/uda.json' \
    --model_cfg='config/bert_base.json'

#- sup_data_dir: 指定训练数据
#- unsup_data_dir: 指定无监督数据位置
#- eval_data_dir: 指定验证数据
#- model_file: 默认为null
#- pretrain_file: 预训练模型，可以为初始化的bert模型，也可以是历史训练版本的模型
#- vocab: bert模型对应vocab词典位置
#- results_dir: 输出模型结果的位置