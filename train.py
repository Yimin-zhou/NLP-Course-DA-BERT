# Copyright 2019 SanghunYun, Korea University.
# (Strongly inspired by Dong-Hyun Lee, Kakao Brain)
# 
# Except load and save function, the whole codes of file has been modified and added by
# SanghunYun, Korea University for UDA.
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

import os
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn

from utils import checkpoint
# from utils.logger import Logger
from tensorboardX import SummaryWriter

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report


class Trainer(object):
    """Training Helper class"""
    def __init__(self, cfg, model, data_iter, optimizer, device):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.device = device

        # data iter    
        if len(data_iter) == 1 and self.cfg.mode != 'pred':
            self.sup_iter = data_iter[0]
        elif len(data_iter) == 1 and self.cfg.mode == 'pred':
            self.pred_iter = data_iter[0]
        elif len(data_iter) == 2 and self.cfg.uda_mode:
            self.sup_iter = self.repeat_dataloader(data_iter[0])
            self.unsup_iter = self.repeat_dataloader(data_iter[1])
        elif len(data_iter) == 2 and not self.cfg.uda_mode:
            self.sup_iter = self.repeat_dataloader(data_iter[0])    # 进行repeat
            self.eval_iter = data_iter[1]
        elif len(data_iter) == 3:
            self.sup_iter = self.repeat_dataloader(data_iter[0])
            self.unsup_iter = self.repeat_dataloader(data_iter[1])
            self.eval_iter = data_iter[2]

#     def train(self, get_loss, get_acc, model_file, pretrain_file):
    def train(self, get_loss, get_f1, model_file, pretrain_file):
        """ train uda"""

        # tensorboardX logging
        if self.cfg.results_dir:
            logger = SummaryWriter(log_dir=os.path.join(self.cfg.results_dir, 'logs'))

        self.model.train()
        self.load(model_file, pretrain_file)    # between model_file and pretrain_file, only one model will be loaded
        model = self.model.to(self.device)
        if self.cfg.data_parallel:                       # Parallel GPU mode
            model = nn.DataParallel(model)

        global_step = 0
        loss_sum = 0.
        max_acc = [0., 0]   # acc, step
        max_f1 = [0., 0]   # f1, step

        # Progress bar is set by unsup or sup data
        # uda_mode == True --> sup_iter is repeated
        # uda_mode == False --> sup_iter is not repeated
        iter_bar = tqdm(self.unsup_iter, total=self.cfg.total_steps) if self.cfg.uda_mode \
              else tqdm(self.sup_iter, total=self.cfg.total_steps)
        print(len(iter_bar))
        for i, batch in enumerate(iter_bar):
            # Device assignment
            if self.cfg.uda_mode:
                sup_batch = [t.to(self.device) for t in next(self.sup_iter)]
                unsup_batch = [t.to(self.device) for t in batch]
            else:
                sup_batch = [t.to(self.device) for t in batch]
                unsup_batch = None

            # update
            self.optimizer.zero_grad()
            final_loss, sup_loss, unsup_loss = get_loss(model, sup_batch, unsup_batch, global_step)
            final_loss.backward()
            self.optimizer.step()

            # print loss
            global_step += 1
            loss_sum += final_loss.item()
            if self.cfg.uda_mode:
                iter_bar.set_description('final=%5.3f unsup=%5.3f sup=%5.3f'\
                        % (final_loss.item(), unsup_loss.item(), sup_loss.item()))
            else:
                iter_bar.set_description('loss=%5.3f' % (final_loss.item()))

            # logging            
            if self.cfg.uda_mode:
                logger.add_scalars('data/scalar_group',
                                    {'final_loss': final_loss.item(),
                                     'sup_loss': sup_loss.item(),
                                     'unsup_loss': unsup_loss.item(),
                                     'lr': self.optimizer.get_lr()[0]
                                    }, global_step)
            else:
                logger.add_scalars('data/scalar_group',
                                    {'sup_loss': final_loss.item()}, global_step)

            if global_step % self.cfg.save_steps == 0:
                self.save(global_step)

            if get_f1 and global_step % self.cfg.check_steps == 0:
                labels, preds = self.eval_f1(get_f1, None, model)
                class_report = classification_report(labels, preds)
                f1 = f1_score(labels, preds)
                acc = accuracy_score(labels, preds)
                precision = precision_score(labels, preds)
                recall = recall_score(labels, preds)
                
                logger.add_scalars('data/scalar_group', {'eval_f1' : f1}, global_step)
                logger.add_scalars('data/scalar_group', {'eval_acc' : acc}, global_step)
                logger.add_scalars('data/scalar_group', {'eval_precision' : precision}, global_step)
                logger.add_scalars('data/scalar_group', {'eval_recall' : recall}, global_step)
                
                if max_f1[0] < f1:
                    self.save(global_step)
                    max_f1 = f1, global_step
                    
                print('Accuracy : %5.3f' % acc)
                print('Precision : %5.3f' % precision)
                print('Recall : %5.3f' % recall)
                print('F1 : %5.3f' % f1)
                print(class_report)
                print('Max F1 : %5.3f Max global_steps : %d Cur global_steps : %d' %(max_f1[0], max_f1[1], global_step), end='\n\n')
                
            if self.cfg.total_steps and self.cfg.total_steps < global_step:
                print('The total steps have been reached')
                print('Average Loss %5.3f' % (loss_sum/(i+1)))
                if get_f1:
                    labels, preds = self.eval_f1(get_f1, None, model)
                    class_report = classification_report(labels, preds)
                    f1 = f1_score(labels, preds)
                    acc = accuracy_score(labels, preds)
                    precision = precision_score(labels, preds)
                    recall = recall_score(labels, preds)

                    logger.add_scalars('data/scalar_group', {'eval_f1' : f1}, global_step)
                    logger.add_scalars('data/scalar_group', {'eval_acc' : acc}, global_step)
                    logger.add_scalars('data/scalar_group', {'eval_precision' : precision}, global_step)
                    logger.add_scalars('data/scalar_group', {'eval_recall' : recall}, global_step)
                    
                    if max_f1[0] < f1:
                        max_f1 = f1, global_step                
                    print('Accuracy : %5.3f' % acc)
                    print('Precision : %5.3f' % precision)
                    print('Recall : %5.3f' % recall)
                    print('F1 : %5.3f' % f1)
                    print(class_report)
                    print('Max F1 : %5.3f Max global_steps : %d Cur global_steps : %d' %(max_f1[0], max_f1[1], global_step), end='\n\n')
                self.save(global_step)
                return
                
        return global_step

    def eval(self, evaluate, model_file, model):
        """ evaluation function """
        if model_file:
            self.model.eval()
            self.load(model_file, None)
            model = self.model.to(self.device)
            if self.cfg.data_parallel:
                model = nn.DataParallel(model)

        results = []
        iter_bar = tqdm(self.sup_iter) if model_file \
            else tqdm(deepcopy(self.eval_iter))
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]

            with torch.no_grad():
                accuracy, result = evaluate(model, batch)
            results.append(result)

            iter_bar.set_description('Eval Acc=%5.3f' % accuracy)
        return results
    
    # added by YuchuanFu on 23 Aug
    def eval_f1(self, evaluate, model_file, model):
        """ evaluation function """
        if model_file:
            self.model.eval()
            self.load(model_file, None)
            model = self.model.to(self.device)
            if self.cfg.data_parallel:
                model = nn.DataParallel(model)

        labels = []
        preds = []
        iter_bar = tqdm(self.sup_iter) if model_file \
            else tqdm(deepcopy(self.eval_iter))
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]

            with torch.no_grad():
                label, pred = evaluate(model, batch)
            for item in label:
                labels.append(item)
            for item in pred:
                preds.append(item)
#             iter_bar.set_description('Eval Acc=%5.3f' % accuracy)
        return labels, preds
    
    # added by YuchuanFu on 28 June
    def pred(self, predict, model_file, model):
        """ prediction function """
        if model_file:
            self.model.eval()
            self.load(model_file, None)
            model = self.model.to(self.device)
            if self.cfg.data_parallel:
                model = nn.DataParallel(model)

        results = []    # 最终results的形式为 [[],[],...,[]] batch的prediction
        neg_prob_all = []
        pos_prob_all = []
        iter_bar = tqdm(self.pred_iter)
        
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad():
                result, neg_prob, pos_prob = predict(model, batch)
                
            for item in result:
                results.append(item.item())
            for score in neg_prob:
                neg_prob_all.append(score.item())
            for score in pos_prob:
                pos_prob_all.append(score.item())

        return results, neg_prob_all, pos_prob_all
    
            
    def load(self, model_file, pretrain_file):
        """ between model_file and pretrain_file, only one model will be loaded """
        if model_file:
            print('Loading the model from', model_file)
            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load(model_file))
            else:
                self.model.load_state_dict(torch.load(model_file, map_location='cpu'))

        elif pretrain_file:
            print('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.ckpt'):  # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'):  # pretrain model file in pytorch
                self.model.transformer.load_state_dict(
                    {key[12:]: value
                        for key, value in torch.load(pretrain_file).items()
                        if key.startswith('transformer')}
                )   # load only transformer parts
    
    def save(self, i):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg.results_dir, 'save')):
            os.makedirs(os.path.join(self.cfg.results_dir, 'save'))
        torch.save(self.model.state_dict(),
                        os.path.join(self.cfg.results_dir, 'save', 'model_steps_'+str(i)+'.pt'))

    def repeat_dataloader(self, iterable):
        """ repeat dataloader """
        while True:
            for x in iterable:
                yield x
