"""Fine-tuning BERT."""
import os
import time
import random
import logging
import pathlib
import argparse
import sklearn
import numpy as np

from functools import partial
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from build_dataset import AmazonProductReviews

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data import (
        TensorDataset,
        DataLoader,
        RandomSampler)

from transformers import (
        BertForSequenceClassification,
        AdamW)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console to logger
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console) 


def get_features_from_batch(batch, device):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    labels = batch['labels'].to(device)

    return input_ids, attention_mask, token_type_ids, labels


class BertForBinaryClassification(nn.Module):
    """BERT for classification task."""
    def __init__(self, args):
        super(BertForBinaryClassification, self).__init__()
        
        self.bert_model = args.bert_model
        self.num_labels = args.num_labels
        self.output_dir = args.output_dir
        self.logging_steps = args.logging_steps 
        self.learning_rate = args.learning_rate
        self.gpu_id = args.gpu_id
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")

        # if end with `.pt` 
        if ".pt" in self.bert_model:
            learner = torch.load(self.bert_model) 
            self.model = learner.model
            logger.info("Loading BERT from meta-learned model")
        else:
            self.model = BertForSequenceClassification.from_pretrained(self.bert_model,
                                                                       num_labels=self.num_labels) 
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self, dataloader):
        self.model.train()
        total_acc, total_count = 0, 0
        start_time = time.time()

        for idx, batch in enumerate(dataloader):
            self.optimizer.zero_grad()
            features = get_features_from_batch(batch=batch,
                                               device=self.device)
            input_ids, attention_mask, token_type_ids, labels = features
            outputs = self.model(input_ids,
                                 attention_mask=attention_mask,
                                 labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

            pred_labels = torch.argmax(outputs[1], dim=1)
            pred_labels_np = pred_labels.detach().cpu().numpy().tolist()

            labels_np = labels.detach().cpu().numpy().tolist()

            acc = accuracy_score(pred_labels_np, labels_np)
            total_acc += acc 
            total_count += 1
            
            if (idx+1) % self.logging_steps == 0 or idx == 0:
                loss_value = loss.item() 
                avg_acc = total_acc / total_count
                msg = "Global step: {:3d}, Loss: {:5.2f}, Training acc: {:5.2f}, Avg. aac: {:8.2f}".format(idx+1,
                                                  loss.item(),
                                                  acc,
                                                  avg_acc)
                logger.info(msg)

                total_acc, total_count = 0, 0 
                start_time = time.time() 
        


    def evaluate(self, dataloader):
        self.model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                features = get_features_from_batch(batch=batch,
                                                   device=self.device)
                input_ids, attention_mask, token_type_ids, labels = features
        
                outputs = self.model(input_ids,
                                     attention_mask=attention_mask,
                                     labels=labels)
                loss = outputs[0]
                pred_labels = torch.argmax(outputs[1], dim=1)
                pred_labels_np = pred_labels.detach().cpu().numpy().tolist()

                labels_np = labels.detach().cpu().numpy().tolist()

                acc = accuracy_score(pred_labels_np, labels_np)
                total_acc += acc 
                total_count += 1

        return total_acc/total_count
