## Example usage
This project are broadly divided into two parts(Fine-tuning, Evaluation).<br/>
**Caution** : <br/>
**Before runing code, you have to check and edit config file** <br/>
**将BERT_Base_Uncased.zip文件解压与当前目录BERT_Base_Uncased文件夹下，作为模型的初始化** <br/>


```
bash download.sh

bash uda.sh
```

## Example usage
1. **激活对应的python环境**
- 首先激活进入对应的python环境

        source torch_3.6/bin/activate
        
        - config/iter/uda.json为利用uda增强进行训练的参数指定
        - config/iter/non_uda.json为常规训练（不利用uda增强进行训练）的参数指定
        - config/iter/pred.json 为模型inference功能的参数指定


2. **运行训练代码**

        python main.py \
            --cfg='config/iter/uda.json' \
            --model_cfg='config/bert_base.json'
        
        - sup_data_dir: 指定训练数据
        - unsup_data_dir: 指定无监督数据位置
        - eval_data_dir: 指定验证数据
        - model_file: 默认为null
        - pretrain_file: 预训练模型，可以为初始化的bert模型，也可以是历史训练版本的模型
        - vocab: bert模型对应vocab词典位置
        - results_dir: 输出模型结果的位置


3. **运行预测代码**
- 运行预测代码，json文件中重要字段含义见下方，得到的预测结果直接输出在config/iter/pred.csv目录下
- 输出的pred.csv文件共有三列，分别为标签、neg_softmax_score、pos_softmax_score

        python main.py \
            --cfg='config/iter/pred.json' \
            --model_cfg='config/bert_base.json'
            
        - model_file: step2中训练好的模型
        - pred_data_dir: 需要预测的数据（仅保留数据一列，且无title）
        - vocab: bert模型对应vocab词典位置




