# Revisiting BPR

**This repository contains the source code for our paper RecSys'24 "Revisiting BPR: A Replicability Study of a Common Recommender System Baseline".**

## Installation

### 1. Dependencies

The instructions were tested on an Ubuntu 22.04 LTS machine with an NVIDIA A100/V100 GPU and a MacBook Pro with M2Pro.

The command below installs all available dependencies.

```bash
poetry install && poetry run pip install cornac==2.0.0
```

There are also extra options that one can turn off. Available options: s3, exp, otherlibs, dev. One can disable them like so:

```bash
poetry install --without s3,otherlibs
```

### 2. Install CLI tools

For MacOS:

```bash
brew install jq miller
```

For Ubuntu:

```bash
apt-get install jq miller
```

### 3. Build an environment for MyMediaLite

```bash
make docker.elliot
```

### 4. Build an environment for Elliot

```bash
make docker.mymedialite
```

## Datasets

| Dataset             | Users  | Items | Actions | Sparsity | Med. User/Item |
| ------------------- | ------ | ----- | ------- | -------- | -------------- |
| Netflix             | 9949   | 4825  | 563577  | 0,9883   | 27/12          |
| ML-20M              | 136677 | 20108 | 9,7M    | 0,9965   | 37/16          |
| MSD                 | 571355 | 41140 | 32,5M   | 0,9986   | 39/383         |
| Yelp                | 252616 | 92089 | 2.2M    | 0,9999   | 5/8            |
| ML-20M (time-split) | 124377 | 12936 | 8.9M    | 0,9944   | 38/57          |

### Netflix

1. Download Netflix Prize dataset from [here](https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz).

2. Extract the dataset from the archive:

```bash
mkdir -p data/netflix \
    && tar xzvf nf_prize_dataset.tar.gz --directory data/netflix --strip-components 1 \
    && tar xf data/netflix/training_set.tar --directory data/netflix
```

3. Combine the files in `training_set/` directory into one file:

```bash
bin/datasets/netflix.sh data/netflix/training_set > data/netflix/full-dataset.csv
```

4. Cutoff users and items that have fewer than 10 actions:

```bash
poetry run python experiments/bpr/cmd/cutoff_samples.py \
    --user-col user --item-col movie \
    --min-users 10 --min-items 10 \
    data/netflix/full-dataset.csv > data/netflix/cutoff-dataset.csv
```

5. Split the dataset into training and validation parts and convert them into JSON Lines format:

```bash
mkdir -p data/netflix/exp && poetry run python experiments/bpr/cmd/split.py \
    --seed 42069 \
    --encoders-dir data/netflix/enc \
    data/netflix/cutoff-dataset.csv \
    data/netflix/exp/train.jsonl data/netflix/exp/eval.jsonl
```

### MovieLens-20M/MSD

1. Prepare datasets following [Revisit iALS](https://arxiv.org/abs/2110.14037) evaluation protocol:

```bash
mkdir -p data \
    && poetry run python experiments/datasets/revisit-ials/generate_data.py --output_dir data/
```

2. Split the dataset into training and validation parts and convert them to JSON Lines format:

```bash
parallel --ungroup 'bin/datasets/format-repro.sh -o data/{}/exp data/{}' ::: ml-20m msd
```

3. Add datasets for autoencoders:

```bash
parallel --ungroup 'bin/datasets/format-repro-multae.sh -o data/{}/exp data/{}/exp' ::: ml-20m msd
```

### MovieLens-20M (time-split)

1. Download the dataset from [here](https://files.grouplens.org/datasets/movielens/ml-20m.zip).

2. Extract the archive using this command:

```bash
mkdir -p data/ml-20m-time-split \
    && unzip ml-20m.zip -d data/ml-20m-time-split \
    && mv data/ml-20m-time-split/ml-20m/* data/ml-20m-time-split \
    && rmdir data/ml-20m-time-split/ml-20m
```

3. Split the dataset into training, validation, and testing parts:

```bash
poetry run python experiments/datasets/time-split/dataset.py \
    data/ml-20m-time-split/ratings.csv \
    data/ml-20m-time-split/processed \
    --user-idx userId --item-idx movieId --value-idx rating --date-idx timestamp
```

4. Convert the dataset to JSON Lines format:

```bash
bin/datasets/format-time-split.sh -o data/ml-20m-time-split/exp data/ml-20m-time-split/processed
```

5. Add dataset for autoencoders:

```bash
bin/datasets/format-time-split-multae.sh -o data/ml-20m-time-split/exp data/ml-20m-time-split/exp
```

### YELP

1. Download the dataset from [here](https://www.yelp.com/dataset).

2. Extract the archive using this command:

```bash
mkdir -p data/yelp && tar xzvf yelp_dataset.tar --directory data/yelp
```

3. Convert the dataset to CSV format:

```bash
mlr --ijsonl --ocsv cut -f 'user_id,business_id,stars,date' \
    data/yelp/yelp_academic_dataset_review.json > data/yelp/reviews.csv
```

4. Split the dataset into training, validation, and testing parts:

```bash
poetry run python experiments/datasets/time-split/dataset.py \
    data/yelp/reviews.csv \
    data/yelp/processed \
    --user-idx user_id --item-idx business_id --value-idx stars --date-idx date --drop-duplicates
```

5. Convert the dataset to JSON Lines:


```bash
bin/datasets/format-time-split.sh -o data/yelp/exp data/yelp/processed
```

6. Add datasets for autoencoders:

```bash
bin/datasets/format-time-split-multae.sh -o data/yelp/exp data/yelp/exp
```

## How to run experiments?

### Scripts

Most experiments were carried out using [experiments/run.py](experiments/run.py) or [experiments/s3_run.py](experiments/s3_run.py).
Both of these scripts support preemptible setup and parallelized hyperparameters search.

### Config

You can find configs for our experiments in the `configs/` directory. All of these files are templated with jinja2. Some of them require their own templated values. For example, to run the PyTorch model, you must inject **train_batch_size** and **embedding_dim** variables using the `--extra-vars` option. One can find these variables in the config. They look like so `{{ ... }}`.
Additionally, the config files for our best models include comments with the best hyperparameters for each dataset they are used with. The comments also include the dataset name and the embedding dimension.

### Hyperparameters Space

The hyperparameter space for each experiment is in the config files in the [configs/](configs/) directory.

### Examples

#### Experiment without hyperparameters search

```bash
poetry run experiments/run.py \
    --seed 13 \
    --extra-vars "dataset=data/netflix/exp;num_users=10000;num_items=5000;embedding_dim=128" \
    configs/RQ1/ours.yaml.j2
```

#### Experiment with hyperparameters search

```bash
poetry run experiments/run.py \
    --name netflix-exp \
    --seed 13 \
    --extra-vars "dataset=data/netflix/exp;num_users=10000;num_items=5000;embedding_dim=128" \
    --search-hp \
    --search-hp-metric auc \
    --search-hp-trials 50 \
    --search-hp-train-best \
    configs/RQ1/ours.yaml.j2
```

#### Parallelized hyperparameters search

First process:

```bash
poetry run experiments/run.py \
    --name netflix-exp \
    --seed 13 \
    --extra-vars "dataset=data/netflix/exp;num_users=10000;num_items=5000;embedding_dim=128" \
    --search-hp \
    --search-hp-storage "{{ database dsn connectioin }}" \
    --search-hp-metric auc \
    --search-hp-trials 50 \
    --search-hp-seed 13 \
    --search-hp-train-best \
    configs/RQ1/ours.yaml.j2

```

Second process:

```bash
poetry run experiments/run.py \
    --name netflix-exp \
    --seed 13 \
    --extra-vars "dataset=data/netflix/exp;num_users=10000;num_items=5000;embedding_dim=128" \
    --search-hp \
    --search-hp-storage "{{ database dsn connectioin }}" \
    --search-hp-metric auc \
    --search-hp-trials 50 \
    --search-hp-seed 14 \
    configs/RQ1/ours.yaml.j2
```

All you need to do is define separate `--search-hp-seed` values for each process. You can run as many parallelized instances as possible to speed up the experiment's process.

#### Preemptible

To enable preemptible for your experiments, add the directory option `-d`.

Scripts that utilize [experiments/s3_run.py](experiments/s3_run.py) have the same structure but require credentials to connect to an S3 bucket.

## Experiments

Below are scripts to train the best models from our experiments.

### Example

A simplified `example.py` script reproduces the best model on the ML-20M dataset with a user-based split.

### Netflix

```bash
poetry run python experiments/run.py \
    configs/RQ1/ours.yaml.j2 \
    -n ours-netflix \
    -d exps/ours-netflix \
    --seed 13 \
    --extra-vars "dataset=data/netflix/exp;num_users=10000;num_items=5000;embedding_dim=64;item_bias=false" \
    --search-hp \
    --search-hp-seed 13 \
    --search-hp-metric auc \
    --search-hp-trials 50 \
    --search-hp-train-best
```

### MovieLens-20M

```bash
poetry run python experiments/run.py \
    configs/RQ2/neg-sampling/ada-sampling-ml-20m.yaml.j2  \
    -n ours-ml-20m-ada-sgd-1024dim \
    -d exps/ours-ml-20m-ada-sgd-1024dim \
    --seed ${SEED:-13} \
    --extra-vars "dataset=data/ml-20m/exp;num_users=136677;num_items=20108;train_batch_size=256;embedding_dim=1024;item_bias=false" \
    --search-hp \
    --search-hp-seed 13 \
    --search-hp-metric ndcg@100 \
    --search-hp-trials 50 \
    --search-hp-train-best
```

### MSD

```bash
poetry run python experiments/run.py \
    configs/RQ2/neg-sampling/ada-sampling-msd.yaml.j2  \
    -n ours-msd-ada-sgd-1024dim \
    -d exps/ours-msd-ada-sgd-1024dim \
    --seed 13 \
    --extra-vars "dataset=data/msd/exp;num_users=571355;num_items=41140;train_batch_size=256;embedding_dim=1024;item_bias=false" \
    --search-hp \
    --search-hp-seed 13 \
    --search-hp-metric ndcg@100 \
    --search-hp-trials 30 \
    --search-hp-train-best
```

### MovieLens-20M (time-split)

```bash
poetry run python experiments/run.py \
    configs/RQ3/time-split/ada-sampling.yaml.j2  \
    -n ours-ml-20m-time-split-ada-256dim \
    -d exps/ours-ml-20m-time-split-ada-256dim \
    --seed 13 \
    --extra-vars "dataset=data/ml-20m-time-split/exp;num_users=124377;num_items=12936;train_batch_size=256;embedding_dim=256;item_bias=false" \
    --search-hp \
    --search-hp-seed 13 \
    --search-hp-metric ndcg@100 \
    --search-hp-trials 50 \
    --search-hp-train-best
```

### YELP

```bash
poetry run python experiments/run.py \
    configs/RQ3/time-split/ada-sampling.yaml.j2  \
    -n ours-yelp-ada-256dim \
    -d exps/ours-yelp-ada-256dim \
    --seed 13 \
    --extra-vars "dataset=data/yelp/exp;num_users=252616;num_items=92089;train_batch_size=256;embedding_dim=256;item_bias=false" \
    --search-hp \
    --search-hp-seed 13 \
    --search-hp-metric ndcg@100 \
    --search-hp-trials 50 \
    --search-hp-train-best
```

## How do you run inference?

When you run a hyperparameters search, each run script ([run.py](experiments/run.py), [s3_run.py](experiments/s3_run.py)) already includes evaluation with the best hyperparameters, controlled through the `--search-hp-train-best` option.
You might need to run a specific [infer.py](experiments/infer.py), [s3_infer.py](experiments/s3_infer.py) to get metrics and raw item logits for each user in the dataset.

```bash
poetry run python experiments/run.py \
    configs/RQ3/time-split/ada-sampling.yaml.j2  \
    -n ours-ml-20m-time-split-ada-256dim \
    -d exps/ours-ml-20m-time-split-ada-256dim \
    --seed 13 \
    --extra-vars "dataset=data/ml-20m-time-split/exp;num_users=124377;num_items=12936;train_batch_size=256;embedding_dim=256;item_bias=false"
```

Additional parameters to support remote storage for Optuna are included. [s3_infer.py](experiments/s3_run.py) script has same options but adds parameters for S3 credentials.

## How do you run a paired t-test?

Infer script generates `user-metrics.jsonl` file, which is used to conduct a paired t-test.

```bash
poetry run python experiments/ttest.py \
  --first <(mlr --ijsonl --ojson cat "{{ your experiment directory }}/user-metrics.jsonl") \
  --second <(mlr --ijsonl --ojson cat "{{ your experiment directory }}/user-metrics.jsonl")
```
