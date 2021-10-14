"""Train meta-learning BERT."""
# The script is modified from `https://github.com/mailong25/meta-learning-bert/blob/master/main.py`

import os
import argparse
import logging
import json
import pathlib
import sys
import time
import random
import numpy as np
import sklearn

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertModel, BertTokenizer

from collections import Counter
from task import MetaTask
from models.maml import Learner

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")

    # Model
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="results/tmp",
                        help="Directory for saving checkpoint and log file.")

    # Training 
    parser.add_argument("--data", type=str, default="dataset.json")
    parser.add_argument("--epochs", type=int, default=3)
    
    parser.add_argument("--outer_batch_size", type=int, default=5,
                        help="Batch size of training tasks")
    parser.add_argument("--inner_batch_size", type=int, default=16,
                        help="Batch size of support set")

    parser.add_argument("--inner_update_step", type=int, default=10)
    parser.add_argument("--inner_update_step_eval", type=int, default=10)
    parser.add_argument("--gpu_id", type=int, default=1)

    # Meta task
    parser.add_argument("--num_domain", type=int, default=100)
    parser.add_argument("--num_support", type=int, default=80,
                        help="Number of support set")
    parser.add_argument("--num_query", type=int, default=20,
                        help="Number of query set")
    
    parser.add_argument("--num_train_task", type=int, default=50,
                        help="Number of meta training tasks")
    parser.add_argument("--num_test_task", type=int, default=3,
                        help="Number of meta testing tasks")

    # Optimizer
    parser.add_argument("--outer_update_lr", type=float, default=5e-5)
    parser.add_argument("--inner_update_lr", type=float, default=5e-5)

    return parser.parse_args()

def get_output_dir(output_dir, file):
    """Joint path for output directory."""
    return pathlib.Path(output_dir,file)


def build_dirs(output_dir, logger):
    """Build hierarchical directories."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info(f"Create folder for output directory: {output_dir}")


def load_train_test_examples(data_file, test_domain, num_domain):
    """Load Amazon customer reviews."""
    train_examples, test_examples = list(), list()
    reviews = json.load(open(data_file))
    mention_domain = [r['domain'] for r in reviews if r["domain"] not in test_domain]
    domain_cnt = Counter(mention_domain)
    num_train_domain = len(domain_cnt)

    if num_domain < num_train_domain:
        select_domain = list()
        for idx, d in enumerate(sorted(domain_cnt.items(), key=lambda kv:int(kv[1]))):
            domain, num_examples = d
            select_domain.append(domain)
            if (idx+1) == num_domain:
                break
    else:
        select_domain = [d for d in domain_cnt]
    
    for review in reviews:
        # Low-resource 
        if review["domain"]  in test_domain:
            test_examples.append(review)

        # High-resource
        elif review["domain"] in select_domain:
            train_examples.append(review)

    return train_examples, test_examples


def task_batch_generator(taskset, is_shuffle, batch_size):
    """Yield a batch of tasks from train or test set."""
    idxs = list(range(0, len(taskset)))
     
    if is_shuffle:
        random.shuffle(idxs)

    for i in range(0, len(idxs), batch_size):
        yield [taskset[idxs[j]] for j in range(i, min(i+batch_size, len(taskset)))]  


def set_random_seed(seed):
    """Set new random seed."""
    torch.backends.cudnn.determinstic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_batch_of_tasks(taskset, is_shuffle = True, batch_size = 4):
    idxs = list(range(0,len(taskset)))
    
    if is_shuffle:
        random.shuffle(idxs)
    for i in range(0,len(idxs), batch_size):
        yield [taskset[idxs[i]] for i in range(i, min(i + batch_size,len(taskset)))]


def main():    
    # Training arguments
    args = get_args()

    # output dir
    output_dir = args.output_dir

    # Logger
    logger = logging.getLogger(__name__)
    build_dirs(output_dir, logger)
    build_dirs(pathlib.Path(output_dir, "ckpt"), logger)
    
    log_file = get_output_dir(output_dir, 'example.log')
    logging.basicConfig(filename=log_file,
                        filemode="w",
                        format="%(asctime)s, %(levelname)s %(message)s",
                        datefmt="%H:%M:%S",
                        level=logging.INFO)

    # Add console to logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    
    logger.info(args)
    # Saving arguments
    write_path = get_output_dir(output_dir, 'hyparams.txt')
    with open(write_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        logger.info(f"Saving hyperparameters to: {write_path}")


    ########## Load dataset ##########
    logger.info("Loading Datasets")
    reviews = json.load(open(args.data))
    
    # Train and test example, 21555 and 300
    test_domains = ["office_products", "automotive", "computer_&_video_games"] 
    train_examples, test_examples = load_train_test_examples(args.data, test_domains, args.num_domain)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    # Meta-Learner
    learner = Learner(args)

    ### Sample testing tasks ###
    test_tasks = MetaTask(test_examples,
                          num_task=args.num_test_task,
                          k_support=args.num_support,
                          k_query=args.num_query,
                          tokenizer=tokenizer,
                          testset=True,
                          test_domain=test_domains)

    global_step = 0
    
    # Train perplexity:  epoch * num_task * ceil(k_support/batch_size) * inner_update_step
    for epoch in range(args.epochs):
        logger.info(f"--- Epoch {epoch+1} ---")
               
        # Build training task set (num_task)
        train_tasks = MetaTask(train_examples,
                               num_task=args.num_train_task,
                               k_support=args.num_support,
                               k_query=args.num_query,
                               tokenizer=tokenizer,
                               testset=False,
                               test_domain=None)

        logger.info(f"Processing {len(train_tasks)} training tasks")

        ### Sample task batch from training tasks ###
        # Sample task batch from total tasks in size of `min(num_task, batch_size)`
        # Each task contains `k_support` + `k_query` examples
        task_batch = create_batch_of_tasks(train_tasks, 
                                           is_shuffle=True,
                                           batch_size=args.outer_batch_size)

        # meta_batch has shape (batch_size, k_support*k_query)
        for step, meta_batch in enumerate(task_batch):
            acc = learner(meta_batch, training=True)
            logger.info(f"Training batch: {step+1}\ttraining accuracy: {acc}\n")
            
            if global_step % 2 == 0:
                # Evaluate Test every 1 batch
                logger.info("--- Evaluate test tasks ---")
                test_accs = list()
                # fixed seed for test task
                set_random_seed(1)
                test_db = task_batch_generator(test_tasks,
                                               is_shuffle=False,
                                               batch_size=1)
            
                for idx, test_batch in enumerate(test_db):
                    acc = learner(test_batch, training=False)
                    test_accs.append(acc)
                    logger.info(f"Testing Task: {idx+1}\taccuracy: {acc}")
            
                logger.info(f"Epoch: {epoch+1}\tTesting batch: {step+1}\tTest accuracy: {np.mean(test_accs)}\n")
            
                # Save model
                pt_file = get_output_dir(args.output_dir, f"ckpt/meta.epoch-{epoch+1}.step-{step+1}.pt")
                torch.save(learner, pt_file)
                logger.info(f"Saving checkpoint to {pt_file}")

                # Reset the random seed
                set_random_seed(int(time.time() % 10))
            
            global_step += 1
            

if __name__ == "__main__":
    main()


