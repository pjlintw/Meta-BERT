"""Perform metatask"""
# The script is modified from https://github.com/mailong25/meta-learning-bert/blob/master/task.py

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import collections
import random
import time
import json, pickle
from torch.utils.data import TensorDataset

LABEL_MAP  = {'positive':0, 'negative':1, 0:'positive', 1:'negative'}

class MetaTask(Dataset):
    def __init__(self, examples, num_task, k_support, k_query, tokenizer, testset, test_domain):
        """

        Args:
          samples: list of samples
          num_task: number of training tasks.
          k_support: number of support sample per task
          k_query: number of query sample per task
          tokenizer: Tokeneizer from `transformers`
          testset: Boolean, whether contruncting test set
          test_domain: List of test domain
        """
        self.examples = examples
        self.num_task = num_task
        self.k_support = k_support
        self.k_query = k_query
        self.tokenizer = tokenizer
        self.testset = testset
        self.test_domain = test_domain
        self.max_seq_length = 256

        self.create_batch(self.num_task, self.testset, self.test_domain)

    def create_batch(self, num_task, testset, test_domain):
        # support set: (k_support, 3)
        # query set: (k_query, 3)
        self.supports = list()
        self.queries = list()

        if testset is True:
            for domain in test_domain:
                domain_examples = [exm for exm in self.examples if exm["domain"]==domain]
            
                assert (self.k_support+self.k_query) <= len(domain_examples)
                domain_train = domain_examples[:self.k_support]
                domain_test = domain_examples[self.k_support:self.k_support+self.k_query]
                
                self.supports.append(domain_train)
                self.queries.append(domain_test)
        else:
            for b in range(num_task):  # for each task
                # 1.select domain randomly

                domain = random.choice(self.examples)['domain']
                domainExamples = [e for e in self.examples if e['domain'] == domain]
            
                # 1.select k_support + k_query examples from domain randomly
                selected_examples = random.sample(domainExamples,self.k_support + self.k_query)
                random.shuffle(selected_examples)
                exam_train = selected_examples[:self.k_support]
                exam_test  = selected_examples[self.k_support:] 
           
                self.supports.append(exam_train)
                self.queries.append(exam_test)

    def create_feature_set(self,examples):
        all_input_ids      = torch.empty(len(examples), self.max_seq_length, dtype = torch.long)
        all_attention_mask = torch.empty(len(examples), self.max_seq_length, dtype = torch.long)
        all_segment_ids    = torch.empty(len(examples), self.max_seq_length, dtype = torch.long)
        all_label_ids      = torch.empty(len(examples), dtype = torch.long)

        for id_,example in enumerate(examples):
            input_ids = self.tokenizer.encode(example['text'])
            attention_mask = [1] * len(input_ids)
            segment_ids    = [0] * len(input_ids)

            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                attention_mask.append(0)
                segment_ids.append(0)

            label_id = LABEL_MAP[example['label']]
            all_input_ids[id_] = torch.Tensor(input_ids).to(torch.long)
            all_attention_mask[id_] = torch.Tensor(attention_mask).to(torch.long)
            all_segment_ids[id_] = torch.Tensor(segment_ids).to(torch.long)
            all_label_ids[id_] = torch.Tensor([label_id]).to(torch.long)

        tensor_set = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids)  
        return tensor_set
    
    def __getitem__(self, index):
        support_set = self.create_feature_set(self.supports[index])
        query_set   = self.create_feature_set(self.queries[index])
        return support_set, query_set

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.num_task
