
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.metrics import confusion_matrix, classification_report

import transformers
from transformers import BertModel, BertForSequenceClassification, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

import logging
import argparse
from tqdm import tqdm

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# disable warnings
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"********** Using : {device} **********")

def train_epoch(model, dataloader_train, dataloader_val, optimizer, scheduler, device, epoch):
    model.train()
    
    loss_train_total = 0
    
    progress_bar = tqdm(dataloader_train, desc="Epoch: {:1d}".format(epoch), leave=False, disable=False)
    
    for batch in progress_bar:
        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2]
        }
        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
    
    tqdm.write('\nEpoch {}'.format(epoch))
    
    loss_train_avg = loss_train_total / len(dataloader_train)
    tqdm.write('Training Loss: {}'.format(loss_train_avg))
    
    val_loss, acc, _ = evaluate(model, dataloader_val, device)
    tqdm.write('Val Loss: {}\Val Accuracy: {}'.format(val_loss, acc))


def flat_accuracy(preds,labels):
    pred_flat = np.argmax(preds,axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)/len(labels_flat)

def evaluate(model, dataloader_val, device):
    model.eval()
    
    loss_val_total = 0
    predictions,true_vals = [],[]

    accuracy_total = 0
    
    for batch in tqdm(dataloader_val):
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':  batch[0],
                  'attention_mask':batch[1],
                  'labels': batch[2]
                 }
        with torch.no_grad():
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total +=loss.item()
        
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
        accracy = flat_accuracy(logits,label_ids)
        accuracy_total += accracy
        
    loss_val_avg = loss_val_total/len(dataloader_val)  
    accracy_avg = accuracy_total/len(dataloader_val)
    
    predictions = np.concatenate(predictions,axis=0)
    true_vals = np.concatenate(true_vals,axis=0) 

    class_report = classification_report(true_vals,np.argmax(predictions,axis=1))

    return loss_val_avg, accracy_avg, class_report

# for letent space
def new_encode(tokens, model_name='bert-base-uncased'):
    # Load pre-trained model
    model = BertModel.from_pretrained(model_name)
    
    # Encode texts
    outputs = model(**tokens)

    # Use the average of the last layer's hidden states as features
    X = outputs.last_hidden_state.mean(dim=1).detach().numpy()

    # Return the input_ids, attention_mask, and BERT embeddings
    features = {
        'input_ids': tokens['input_ids'],
        'attention_mask': tokens['attention_mask'],
        'bert_output': X
    }

    return features

def generate_synthetic_samples(features, labels, k=1, lambd=0.5):
    X = features['bert_output']
    input_ids = features['input_ids']
    attention_mask = features['attention_mask']

    n_samples, n_features = X.shape
    # Ensure k is less than the number of samples
    k = min(k, n_samples - 1)
    
    # Initialize array for synthetic samples
    X_synthetic = np.empty((n_samples * k, n_features))
    labels_synthetic = [0] * (n_samples * k)
    input_ids_synthetic = np.empty((n_samples * k, input_ids.shape[1]))
    attention_mask_synthetic = np.empty((n_samples * k, attention_mask.shape[1]))

    for i in range(n_samples):
        # Find the k nearest neighbors of sample i
        distances = np.linalg.norm(X - X[i], axis=1)
        # Get the indices of the k nearest neighbors
        neighbors = np.argpartition(distances, k)[:k]
        # Exclude the sample itself
        neighbors = neighbors[neighbors != i]

        # Generate synthetic samples
        for j in range(k - 1):
            X_synthetic[i*k + j] = (X[i] - X[neighbors[j]]) * lambd + X[i]
            labels_synthetic[i*k + j] = labels[i]
            input_ids_synthetic[i*k + j] = input_ids[i]
            attention_mask_synthetic[i*k + j] = attention_mask[i]

    return {
        'bert_output': X_synthetic,
        'input_ids': input_ids_synthetic,
        'attention_mask': attention_mask_synthetic
    }, labels_synthetic



def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Description of your script')

    # Add arguments
    parser.add_argument('--train_data', type=str, help='Data path of training data')
    parser.add_argument('--valid_data', type=str, help='Data path of validation data')
    parser.add_argument('--test_data', type=str, help='Data path of test data')

    parser.add_argument('--model', type=str, default='bert-base-uncased', 
                        help='Model path of the model')
    
    parser.add_argument('--max_len', type=int, default=512, 
                        help='Maximum length of the input sequence')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='Number of training epochs.')
    parser.add_argument('--train_batch', type=int, default=64,
                        help='Batch size of training data')
    parser.add_argument('--test_batch', type=int, default=64,
                        help='Batch size of test data')

    parser.add_argument('--output', type=str, help='Output path of the model')


    # Parse the command-line arguments
    args = parser.parse_args()

    logger.info("***** 1. Read the raw data from csv file. *****")
    train = pd.read_csv(args.train_data)
    valid = pd.read_csv(args.valid_data)
    test = pd.read_csv(args.test_data)

    training_sentences = train["text"].values.tolist()
    training_labels = train["score"].tolist()

    validation_sentences = valid["text"].values.tolist()
    validation_labels = valid["score"].tolist()

    test_sentences = test["text"].values.tolist()
    test_labels = test["score"].tolist()

    logger.info("***** 2. Tokenize the data. *****")
    
    tokenizer = BertTokenizer.from_pretrained(args.model)

    encoder_train = tokenizer.batch_encode_plus(training_sentences,
                                            add_special_tokens = True,
                                            return_attention_masks = True,
                                            pad_to_max_length = True,
                                            max_length = args.max_len,
                                            return_tensors = 'pt')

    encoder_val = tokenizer.batch_encode_plus(validation_sentences,
                                            add_special_tokens = True,
                                            return_attention_masks = True,
                                            pad_to_max_length = True,
                                            max_length = args.max_len,
                                            return_tensors = 'pt')
                                            
    encoder_test = tokenizer.batch_encode_plus(test_sentences,
                                            add_special_tokens = True,
                                            return_attention_masks = True,
                                            pad_to_max_length = True,
                                            max_length = args.max_len,
                                            return_tensors = 'pt')

    logger.info("***** 3. Convert the data to PyTorch tensors and latent space. *****")

    text = new_encode(encoder_train)
    latent_encoder_train, train_labels = generate_synthetic_samples(text, training_labels, k=4, lambd=0.5)
    
    input_ids_train = torch.tensor(latent_encoder_train['input_ids']).long()
    attention_masks_train = torch.tensor(latent_encoder_train["attention_mask"]).long()
    labels_train = torch.tensor(train_labels)

    input_ids_val = encoder_val['input_ids']
    attention_masks_val = encoder_val["attention_mask"]
    labels_val = torch.tensor(validation_labels)

    input_ids_test = encoder_test['input_ids']
    attention_masks_test = encoder_test["attention_mask"]
    labels_test = torch.tensor(test_labels)
    
    print(f"Training: {input_ids_train.shape}")
    print(f"attention_masks_train: {attention_masks_train.shape}")
    print(f"labels_train: {labels_train.shape}")

    data_train = TensorDataset(input_ids_train,attention_masks_train,labels_train)
    data_val = TensorDataset(input_ids_val,attention_masks_val,labels_val)
    data_test = TensorDataset(input_ids_test,attention_masks_test,labels_test)

    
    dataloader_train = DataLoader(data_train, batch_size = args.train_batch)
    dataloader_val = DataLoader(data_val, sampler= RandomSampler(data_val), batch_size = args.test_batch)
    dataloader_test = DataLoader(data_test, sampler= SequentialSampler(data_test), batch_size = args.test_batch)


    logger.info("***** 4. Initialize the model and optimizer. *****")
    model = BertForSequenceClassification.from_pretrained(args.model,
                                        num_labels = 2,
                                        output_attentions = False,
                                        output_hidden_states =  False).to(device)
    optimizer = AdamW(model.parameters(),lr = 1e-5, eps = 1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, \
                                                num_training_steps = len(dataloader_train)*args.epochs)
    
    logger.info("***** 5. Train the model. *****")
    for epoch in range(args.epochs):
        train_epoch(model, dataloader_train, dataloader_val, optimizer, scheduler, device, epoch)
    
    logger.info("***** 6. Save the model. *****")
    ouput_dir = args.output + '/model_save/'
    if not os.path.exists(ouput_dir):
        os.makedirs(ouput_dir)

    model_to_save = model.module if hasattr(model,'module') else model
    model_to_save.save_pretrained(ouput_dir)

    _, acc, class_report = evaluate(model, dataloader_test, device)    
    
    print(class_report)

    # save the classification report
    logger.info("***** 7. Save the classification report. *****")
    with open(os.path.join(args.output, "classification_report.txt"), "w") as f:
        f.write(class_report)

if __name__ == "__main__":
    main()


