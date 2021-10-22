# Meta-BERT: Learning to Learn fast For Low-Resource Text Classification

[**Installation**](#installation) | [**Data**](#dataset) | [**Meta-BERT**](#meta-bert)

We implement Meta-BERT that trains BERT for fast adapting to new tasks using limited examples. The distinctive feature of Meta-BERT is to meta-train the BERT model using First-Order-MAML, FOMAML, which enables the model to learn new tasks quickly. 

The proposed algorithm is a model-agnostic, in the sense that it can be directly applied to any BERT architecture. We investigate the capacity to adapt Meta-BERT and BERT models to the unseen tasks in which only a limited set of task-specific data are available. We demonstrate that our Meta-BERT outperforms BERT in low-resource fine-tuning by +2.2% accuracy improvement. For zero-shot classification, Meta-BERT achieves +5.7% improvement compared to BERT model. 

* New October 13th, 2021: Meta-learning with BERT

## Files Structure

We implement our Meta-BERT, including the procedures for meta-training and fine-tuning. We explains the purpose of these files.

The figure below uses a `results/meta.nt10-.nd-5/` as the output directory. When specifying path to the flag `--output_dir` for saving results, the hyperparameters, checkpoints and log files will be saved in the folder. For instance, the example shows these reuslt files saved in the `results/meta.nt10-.nd-5/` folder.

Note that `#S` refers to a number of the step in script and `#E` for the number of epoch.

```
|-- models
|   └── maml.py                           # First-Order MAML
|-- results
|   └── meta.nt-10.nd-5
|       |-- example.log                   # Log file 
|       |-- hyparams.txt                  # Hyperparameters
|       └── ckpt
|           └── meta.epoch-#E.step-#S.pt  # Checkpoint for the Meta-BERT saved at the #S-th step of #E-th epoch
|-- dataset.json                          # Subset of Amazon Custom Reviews
|-- task.py                               # Meta tasks builder script
|-- build_dataset.py                      # Fine-tuning data builder script
|-- run_meta_training                     # Meta-training script
└── run_finetuning                        # Fine-tining script
```


## Installation

### Python version

* Python >= 3.8

### Environment

Create an environment from file and activate the environment.

```
conda env create -f environment.yml
conda activate meta-bert
```

If conda fails to create an environment from `environment.yml`. This may be caused by the platform-specific build constraints in the file. Try to create one by installing the important packages manually. The `environment.yml` was built in Linux.

**Note**: Running `conda env export > environment.yml` will include all the 
dependencies conda automatically installed for you. Some dependencies may not work in different platforms.
We suggest you to use the `--from-history` flag to export the packages to the environment setting file.
Make sure `conda` only exports the packages that you've explicitly asked for.

```
conda env export > environment.yaml --from-history
```

## Dataset

We use a subset of `Amazon Customer Reviews` dataset which contains 21,855 examples for sentiment analysis. Here, we provide the deitialed set-ups for building the datasets for meta-learning and fine-tuning.

### Dataset Preprocessing

The subset of `Amazon Customer Reviews` dataset contains 21,855 reviews from 22 product types. We refer the product type as the domain representing the training task in the meta-learning.

We first select 300 examples from 3 domains: `Automotive`, `Computer & Video Games` and `Office Produces` as the test set. Each has only 100 data samples. The test split is suitable to serve as the unseen tasks for meta-learning and as the evaluation of the domain adaptation that the model learns to effectively transfer prior knowledge into new domain. We select `D = {1, 3, 5, 19}` domain(s) for building the low-resource and high-resource training splits. These four training splits contain 698, 2512, 4485, 22555 data examples from 3, 5 and 19 domains respectively. 

We refer the examples collected from 1 domain as low-resource training set and the examples from 3, 5 and 19 domains as high-resource splits. For detailed description, please read the paper.

Note that the training examples are selected from different domains of the test set. We evaluate the Meta-BERT on the unseen test set for better understanding its generalization performance. The subset of dataset can be found in `dataset.json`.

<!--We select the examples by choosing the domain with minimum examples. For example, the low- resource training split uses the data points from 1 domain, Cell phones & Services, which has 698 reviews. Following the criteria, the three high-resource splits contain 2512, 4485, 21555 data examples from 3, 5 and 19 domains respectively.
-->

## Meta-BERT

Meta-BERT's model architecture is identical to BERT with a classifier layer where it is meta-trained for fast adpation. The complete learning strategy consists of two stages: **mete-training** and **fine-tuning**. To reproduce our experiements, please refer to the paper.

### Meta-Training

In the meta-training stage, the pre-trained BERT’s parameters is trained using FOMAML and to be able to learn the unseen domain quickly. It enables the model to learn a set of initial parameters that adapts to new tasks quickly with limited examples. 

To meta-train BERT with FOMAML, execute `run_meta_training.py` with the flag `--output_dir` to specify the repository for saving `hyparams.txt`,  ckeckpoints and log file.

```python
python run_meta_training.py \
 --output_dir results/meta-train-tmp
```

The program will use the flasgs:
`num_train_task=50`, `num_test_task=3`, `num_domain=100`, `num_support=80`, `num_query=20` and save all the results under the `results/meta-train-tmp` folder.

We provide a brief description of meta-learning algorithm which is involved with the flags above: We sample `num_train_task` from distributions over training tasks for explicitly optmizing the model's initial parameters. The flag `num_domain` decides how many domains are chosen to construct the distribution. When `num_domain` exceeds the number of the available domains, we choose all the domains. Follow the same procedure, `num_test_task` determines the number of testing tasks. Each task contains `num_support * num_query` examples which are used for the gradient computation and evalution respectively.

Note that the checkpoints of Meta-BERT will under the `results/meta-train-tmp/ckpt`. You need to provide Meta-BERT's checkpoint to fine-tune it using the script below.


### Fine-Tuning Meta-BERT / BERT

In the fine-tuning stage, the model is initialized with the parameters learnt by meta-training or the parameters from standard BERT. We conduct the experiemets for both Meta-BERT and vanilla BERT which is without meta-training in low-resource and high-resource settings. 

To fune-tune the model, use the flag `--bert_model` to choose which BERT you fine-tune. The default value of `--bert_model` uses `bert-base-uncase`. You can fine-tune Meta-BERT by specifying the path of the ckeckpoint. For example:

```python
python run_finetuning.py \
 --bert_model  results/meta-train-tmp/ckpt/meta.epoch-1.step-1.pt \
 --output_dir results/fine-tuning-tmp \
 --epochs 5
```

The script will initialize the model from the checkpoint `meta.epoch-1.step-1.pt` and run 5 epochs. The fine-tuning results will be saved in `results/fine-tuning-tmp` folder. 

### Zero-Shot Classification

To perform zero-shot classification, simply run `run_fine-tuning.py` with the flag `zero_shot`. Without passing checkpoint to `bert_model` flag, the model is initialized from `bert-base-uncased`.   

```python
python run_finetuning.py \
  --output_dir results/zero-shot-tmp \
  --zero_shot True
```

After running the program, `hyparams.txt`, checkpoints and log file will be saved in `results/zero-shot-tmp` folder. 

### Contact Information

For help or issues using our code, please submit a GitHub issue.




