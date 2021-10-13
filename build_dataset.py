"""Build dataset for fine-tuning BERT."""
import json
import random
from transformers import BertModel, BertTokenizer
from task import MetaTask

import torch

# Use these3 categories for testing 
test_domain = ["office_products", "automotive", "computer_&_video_games"]


class AmazonProductReviews(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def partition_examples(examples, test_domains):
    """Partition train and test examples according to test domains
    
    Args:
      examples: list of examples, each example is dict
                containing `text`, `label` and `domain `as keys.
      test_domains: list of domain name (str) for testing.  
    
    Returns:
      train_examples: List of examples which are not in test domains.
      test_examples: List of examples which are in test domains.
    """
    domain_str = "\t".join(test_domain)
    print(f"Split examples into train and test sets by test domain: {domain_str}")
    
    train_lst, test_lst = list(), list()
    for exm in examples:
        if exm["domain"] not in test_domains:
            train_lst.append(exm)
        else:
            test_lst.append(exm)
    return train_lst, test_lst


def main():
    data_file = "dataset.json"

    reviews = json.load(open(data_file))

    # Label2idx
    label2idx = {"positive": 1, "negative": 0}

    # List of examples
    train_set, test_set =  partition_examples(reviews, test_domain)

    train_texts = [exm["text"] for exm in train_set ]
    train_lables = [label2idx[exm["label"]] for exm in train_set]
    test_texts = [exm["text"] for exm in test_set]
    test_labels = [label2idx[exm["label"]] for exm in test_set]

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",  do_lower_case=True)
    
    # Encoding
    text_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    

    # test_texts, test_labels = read_imdb_split("aclImdb/test")

    # print(len(train_set))
    # print(len(test_set))
    # print(test_set)
    
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    #test = MetaTask(test_set, 
    #                num_task=3,
    #                k_support=80,
    #                k_query=20, 
    #                tokenizer=tokenizer,
    #                testset=True,
    #                test_domain=test_domain)

    # size (num_task, supports)
    # print(len(test.supports[0]), len(test.supports[1]), len(test.supports[2]))
    # print(len(test.queries[0]), len(test.queries[1]), len(test.queries[2]))
   
    # for task_batch in task_batch_generator(test, False, 1):
    #    print(task_batch)
    #    print("\n\n")
    

if __name__ == "__main__":
    main()
