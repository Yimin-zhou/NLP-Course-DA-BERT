# Copyright 2019 SanghunYun, Korea University.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fire
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import train
from load_data import load_data
from utils.utils import set_seeds, get_device, _get_device, torch_device_one
from utils import optim, configuration

from sklearn.metrics import f1_score, classification_report
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
        
# TSA
def get_tsa_thresh(schedule, global_step, num_train_steps, start, end):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    output = threshold * (end - start) + start
    return output.to(_get_device())


def main(cfg, model_cfg):
    # Load Configuration
    cfg = configuration.params.from_json(cfg)                   # Train or Eval cfg
    model_cfg = configuration.model.from_json(model_cfg)        # BERT_cfg
    set_seeds(cfg.seed)

    # Load Data & Create Criterion
    data = load_data(cfg)
    if cfg.uda_mode:
        unsup_criterion = nn.KLDivLoss(reduction='none')
        data_iter = [data.sup_data_iter(), data.unsup_data_iter()] if cfg.mode=='train' \
            else [data.sup_data_iter(), data.unsup_data_iter(), data.eval_data_iter()]  # train_eval
    elif cfg.mode == 'pred':
        data_iter = [data.pred_data_iter()]
    else:
#         data_iter = [data.sup_data_iter()]
        data_iter = [data.sup_data_iter(), data.eval_data_iter()] # edited by YuchuanFu on 6th, July
    sup_criterion = nn.CrossEntropyLoss(reduction='none')
#     sup_criterion = FocalLoss()
#     sup_criterion = F.FocalLoss()
    
    # Load Model
    model = models.Classifier(model_cfg, len(data.TaskDataset.labels))

    # Create trainer
    trainer = train.Trainer(cfg, model, data_iter, optim.optim4GPU(cfg, model), get_device())

    # Training
    def get_loss(model, sup_batch, unsup_batch, global_step):

        # logits -> prob(softmax) -> log_prob(log_softmax)

        # batch
        input_ids, segment_ids, input_mask, label_ids = sup_batch
        if unsup_batch:
            ori_input_ids, ori_segment_ids, ori_input_mask, \
            aug_input_ids, aug_segment_ids, aug_input_mask  = unsup_batch

            input_ids = torch.cat((input_ids, aug_input_ids), dim=0)
            segment_ids = torch.cat((segment_ids, aug_segment_ids), dim=0)
            input_mask = torch.cat((input_mask, aug_input_mask), dim=0)
            
        # logits
        logits = model(input_ids, segment_ids, input_mask)

        # sup loss
        sup_size = label_ids.shape[0]            
        sup_loss = sup_criterion(logits[:sup_size], label_ids)  # shape : train_batch_size
        if cfg.tsa:
            tsa_thresh = get_tsa_thresh(cfg.tsa, global_step, cfg.total_steps, start=1./logits.shape[-1], end=1)
            larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh   # prob = exp(log_prob), prob > tsa_threshold
            # larger_than_threshold = torch.sum(  F.softmax(pred[:sup_size]) * torch.eye(num_labels)[sup_label_ids]  , dim=-1) > tsa_threshold
            loss_mask = torch.ones_like(label_ids, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
            sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch_device_one())
        else:
            sup_loss = torch.mean(sup_loss)

        # unsup loss
        if unsup_batch:
            # ori
            with torch.no_grad():
                ori_logits = model(ori_input_ids, ori_segment_ids, ori_input_mask)
                ori_prob   = F.softmax(ori_logits, dim=-1)    # KLdiv target
                # ori_log_prob = F.log_softmax(ori_logits, dim=-1)

                # confidence-based masking
                if cfg.uda_confidence_thresh != -1:
                    unsup_loss_mask = torch.max(ori_prob, dim=-1)[0] > cfg.uda_confidence_thresh
                    unsup_loss_mask = unsup_loss_mask.type(torch.float32)
                else:
                    unsup_loss_mask = torch.ones(len(logits) - sup_size, dtype=torch.float32)
                unsup_loss_mask = unsup_loss_mask.to(_get_device())
                    
            # aug
            # softmax temperature controlling
            uda_softmax_temp = cfg.uda_softmax_temp if cfg.uda_softmax_temp > 0 else 1.
            aug_log_prob = F.log_softmax(logits[sup_size:] / uda_softmax_temp, dim=-1)

            # KLdiv loss
            """
                nn.KLDivLoss (kl_div)
                input : log_prob (log_softmax)
                target : prob    (softmax)
                https://pytorch.org/docs/stable/nn.html

                unsup_loss is divied by number of unsup_loss_mask
                it is different from the google UDA official
                The official unsup_loss is divided by total
                https://github.com/google-research/uda/blob/master/text/uda.py#L175
            """
            unsup_loss = torch.sum(unsup_criterion(aug_log_prob, ori_prob), dim=-1)
            unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1), torch_device_one())
            final_loss = sup_loss + cfg.uda_coeff*unsup_loss

            return final_loss, sup_loss, unsup_loss
        return sup_loss, None, None

    # evaluation
    def get_acc(model, batch):
        # input_ids, segment_ids, input_mask, label_id, sentence = batch
        input_ids, segment_ids, input_mask, label_id = batch
        logits = model(input_ids, segment_ids, input_mask)
        _, label_pred = logits.max(1)
#         print(len(label_pred))   # eval时展示预测label
#         print(label_pred)  # edited by YuchuanFu on 28 June
        
        print(label_pred)
        print(label_id)
        result = (label_pred == label_id).float()
        accuracy = result.mean()
        # output_dump.logs(sentence, label_pred, label_id)    # output dump

        return accuracy, result
    
    # evaluation - f1 score
    # added by YuchuanFu on 19 Aug
    def get_f1(model, batch):
        
        input_ids, segment_ids, input_mask, label_id = batch
        logits = model(input_ids, segment_ids, input_mask)
        _, label_pred = logits.max(1)
        labels = label_id.cpu().numpy()
        preds = label_pred.cpu().numpy()
        
        return labels, preds
    
    # edited by YuchuanFu on 28 June
    # prediction
    def get_pred(model, batch):
        
        input_ids, segment_ids, input_mask = batch
        logits = model(input_ids, segment_ids, input_mask)
        _, label_pred = logits.max(1)
        # logits shape: [16, 2]
        softmax_score = torch.softmax(logits, 1)
        neg_prob = softmax_score[:, 0]
        pos_prob = softmax_score[:, 1]

        return label_pred, neg_prob, pos_prob
    

    if cfg.mode == 'train':
        trainer.train(get_loss, None, cfg.model_file, cfg.pretrain_file)

    if cfg.mode == 'train_eval':
#         trainer.train(get_loss, get_acc, cfg.model_file, cfg.pretrain_file)
        trainer.train(get_loss, get_f1, cfg.model_file, cfg.pretrain_file)

    if cfg.mode == 'eval':
#         results = trainer.eval(get_acc, cfg.model_file, None)
#         total_accuracy = torch.cat(results).mean().item()
#         print('Accuracy :' , total_accuracy)
        labels, preds = trainer.eval_f1(get_f1, cfg.model_file, None)        
        f1 = f1_score(labels, preds)
        class_report = classification_report(labels, preds)
        print('F1 :', f1)
        print(class_report)

    # edited by YuchuanFu on 28 June
    if cfg.mode == 'pred':
        results, neg_prob_all, pos_prob_all = trainer.pred(get_pred, cfg.model_file, None) # 形式为 [[],[],...,[]]的prediction
        results = pd.DataFrame([results, neg_prob_all, pos_prob_all]).T
        results.columns = ['pred', 'neg_softmax_score', 'pos_softmax_score']
        results.to_csv('config/iter/pred.csv', index=0)    # 方便下载

if __name__ == '__main__':
    fire.Fire(main)
