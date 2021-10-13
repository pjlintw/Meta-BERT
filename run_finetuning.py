"""Fine-tuning BERT from bert-base-uncased or meta-leart weights."""
import os
import sys
import json
import time
import random
import logging
import pathlib
import argparse
import sklearn
import numpy as np

from collections import Counter
from functools import partial
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from build_dataset import AmazonProductReviews

import torch
from torch.nn import functional as F
from torch.utils.data import (
        TensorDataset,
        DataLoader,
        RandomSampler)
from transformers import BertTokenizer
from models.bert import BertForBinaryClassification



def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")

    # Model
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="results/transfer",
                        help="Directory for saving checkpoint and log file.")

    # Training 
    parser.add_argument("--data", type=str, default="dataset.json")
    parser.add_argument("--epochs", type=int, default=1) 
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--num_domain", type=int, default=50)
    parser.add_argument("--gpu_id", type=int, default=0)

    # Zero shot
    parser.add_argument("--zero_shot", type=bool, default=False)

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=5e-5)

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


def set_random_seed(seed):
    """Set new random seed."""
    torch.backends.cudnn.determinstic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

   
def main():    
    # Training arguments
    args = get_args()

    SEED = 123

    # output dir
    output_dir = args.output_dir

    # Logger
    logger = logging.getLogger(__name__)
    build_dirs(output_dir, logger)
    build_dirs(pathlib.Path(output_dir, "ckpt"), logger)
    
    log_file = get_output_dir(output_dir, 'example.log')
    logging.basicConfig(filename=log_file,
                        filemode="w",
                        format="%(asctime)s, %(msecs)d %(name)s %(levelname)s %(message)s",
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

    # Label2idx
    label2idx = {"positive": 1, "negative": 0}

    train_texts = [exm["text"] for exm in train_examples ]
    train_labels = [label2idx[exm["label"]] for exm in train_examples]
    test_texts = [exm["text"] for exm in test_examples]
    test_labels = [label2idx[exm["label"]] for exm in test_examples]

    # Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts,
                                                                        train_labels,
                                                                        test_size=.2,
                                                                        random_state=SEED)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",  do_lower_case=True)
    
    # Encoding
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
 
    # Datasets
    train_dataset = AmazonProductReviews(train_encodings, train_labels)
    val_dataset = AmazonProductReviews(val_encodings, val_labels)
    test_dataset = AmazonProductReviews(test_encodings, test_labels)

    # Train, valid and test dataloader
    dataLoader_fn = partial(DataLoader,
                            batch_size=args.batch_size,
                            num_workers=2)


    train_loader = dataLoader_fn(train_dataset, shuffle=True)
    val_loader = dataLoader_fn(val_dataset, shuffle=True)
    test_loader = dataLoader_fn(test_dataset, shuffle=False)

    ### Model ###
    model = BertForBinaryClassification(args)


    # Perform zero shot
    if args.zero_shot:
        # Evaluate on test set
        acc_test = model.evaluate(test_loader)
        logger.info("test accuracy {:8.3f}".format(acc_test))

        # Save model
        pt_file = get_output_dir(args.output_dir, f"ckpt/transfer.pt")
        torch.save(model, pt_file)
        logger.info(f"Saving checkpoint to {pt_file}")

        return None

    for epoch in range(1, args.epochs+1):    
        epoch_start_time = time.time()
        model.train(train_loader)
        acc_val = model.evaluate(val_loader)

        logger.info("\n{}\n| end of epoch {:3d} | time: {:5.2f}s | "
                    "valid accuracy {:8.3f} \n{}".format("-"*59,
                                                     epoch,
                                                     time.time() - epoch_start_time,
                                                     acc_val,
                                                     "-"*59))
        
        # Evaluate on test set
        acc_test = model.evaluate(test_loader)
        logger.info("test accuracy {:8.3f}".format(acc_test))

        # Save model
        pt_file = get_output_dir(args.output_dir, f"ckpt/transfer.epoch-{epoch}.pt")
        torch.save(model, pt_file)
        logger.info(f"Saving checkpoint to {pt_file}")

       
if __name__ == "__main__":
    main()


