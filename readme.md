SpherePair

```markdown
# SpherePair: An Anchor-Free Approach to Deep Constrained Clustering

This repository contains the implementation of **SpherePair** (proposed in our paper *"SpherePair: An Anchor-Free Approach to Deep Constrained Clustering"*) and various baseline models for deep constrained clustering (DCC). With the provided scripts, you can run experiments on datasets such as MNIST, FashionMNIST, CIFAR-10, CIFAR-100, STL-10, ImageNet-10, and Reuters, using either balanced or imbalanced constraint sets. You can also switch between pre-trained versus non-pretrained modes, or run different baseline methods.

---

## Table of Contents

1. [Project Structure](#project-structure)  
2. [Installation](#installation)  
3. [Basic Usage](#basic-usage)  
4. [SpherePair Demo on FashionMNIST](#spherepair-demo-on-fashionmnist)  
   - [Balanced Constraints + Pretrain](#41-balanced-constraints--pretrain)  
   - [Balanced Constraints + No Pretrain](#42-balanced-constraints--no-pretrain)  
   - [Imbalanced Constraints + Pretrain](#43-imbalanced-constraints--pretrain)  
   - [Imbalanced Constraints + No Pretrain](#44-imbalanced-constraints--no-pretrain)  
5. [Extended Experiments](#extended-experiments)  
6. [Hyperparameter Tuning (AutoEmbedder & VolMaxDCC)](#hyperparameter-tuning-autoembedder--volmaxdcc)  
7. [Where to Find Logs/Results](#where-to-find-logsresults)  
8. [FAQ](#faq)  
9. [Citation](#citation)  
10. [License](#license)

---

## 1. Project Structure

Below is a simplified overview of the core folders and scripts:

```
.
├── experiment
│   ├── run_model_Sphere_Kmeans.py
│   ├── run_model_Sphere_Hierarchical.py
│   ├── run_model_AutoEmbedder_Kmeans.py
│   ├── run_model_AutoEmbedder_Hierarchical.py
│   ├── run_model_CIDEC.py
│   ├── run_model_SDEC.py
│   ├── run_model_VanillaDCC.py
│   ├── run_model_VolMaxDCC.py
│   ├── run_model_Sphere_Kmeans_model_selection.py
│   ├── tool_pretrain_sdae.py
│   ├── tool_pretrain_sphere.py
│   ├── tool_pretrain_sphere_model_selection.py
│   ├── tool_createCons.py
│   ├── tool_tSNE.py
│   ├── job_demo_sphere
│   └── dataset
│       ├── cifar10/
│       ├── cifar100/
│       ├── fashion_mnist/ (MNIST-like format)
│       ├── imagenet10/
│       ├── mnist/ (MNIST-like format)
│       ├── reuters/
│       └── stl10/
├── lib
│   ├── consRules.py
│   ├── datasets.py
│   ├── dec.py
│   ├── denoisingAutoencoder.py
│   ├── denoisingAutoencoderSphere.py
│   ├── loadData.py
│   ├── ops.py
│   ├── stackedDAE.py
│   ├── stackedDAESphere.py
│   └── utils.py
└── model
    ├── model_SpherePairs.py
    ├── model_AutoEmbedder.py
    ├── model_CIDEC.py
    ├── model_SDEC.py
    ├── model_VanillaDCC.py
    └── model_VolMaxDCC.py
```

- **experiment/**:  
  - `run_model_*.py`: scripts to run a specific model (SpherePair or baselines) with a chosen clustering method (KMeans or hierarchical), optionally using pretraining.  
  - `tool_pretrain_*.py`: scripts for autoencoder pretraining (either Sphere-specific or general SDAE).  
  - `tool_createCons.py`: creates constraint files (balanced or imbalanced) for training/testing sets.  
  - `tool_tSNE.py`: utility for t-SNE visualization.  
  - `job_demo_sphere`: optional folder for job submission or demo scripts.  
  - `dataset/`: contains the data files or pre-processed pickle/pt files.

- **lib/**: library code for handling datasets, constraints, ops, autoencoders, etc.

- **model/**: model definitions for SpherePair and baseline methods (AutoEmbedder, CIDEC, SDEC, VanillaDCC, VolMaxDCC).

---

## 2. Installation

We will provide an `environment.yml` file to create the Conda environment. Make sure you have [conda](https://docs.conda.io/en/latest/) or [mamba](https://github.com/mamba-org/mamba) installed. Then:

```bash
conda env create -f environment.yml
conda activate SpherePairEnv
```

> At the moment, the `environment.yml` file is not fully ready, but it will be added soon.

---

## 3. Basic Usage

A typical workflow for DCC experiments includes:

1. (Optional) **Pretrain** an autoencoder if the model version requires it.  
2. **Generate** constraint files (balanced or imbalanced).  
3. **Run** the desired model script to train and evaluate clustering performance.

General model versions:

- **SpherePair** with KMeans or hierarchical clustering, either with or without pretraining.  
- **AutoEmbedder**, **CIDEC**, **SDEC**, **VanillaDCC**, **VolMaxDCC**.  

You can specify parameters such as the dataset name, constraint rule (`balance` vs `extraCLs`), constraint size, whether to use pretraining, etc.

---

## 4. SpherePair Demo on FashionMNIST

Here we show examples of running SpherePair on FashionMNIST (`fmnist`) under different scenarios: balanced vs. imbalanced constraints, and with vs. without pretraining. These commands can be adapted to other datasets or models as needed.

### 4.1 Balanced Constraints + Pretrain

1. **Pretrain** the SDAE for SpherePair:
   ```bash
   python tool_pretrain_sphere.py --dataset "fmnist"
   # default --dim=10
   ```

2. **Generate** 10 incrementally sized balanced constraint sets (from 1k to 10k) for the train set, and 1k for the test set:
   ```bash
   python tool_createCons.py --dataset "fmnist" \
       --consRule "balance" \
       --set "train" \
       --orig_num "1000" --extra_num "9000" \
       --J "10" \
       --imbCluster "0" \
       --modelVersion "Sphere_Kmeans_Pretrain" \
       --expName "demo"

   python tool_createCons.py --dataset "fmnist" \
       --consRule "balance" \
       --set "test" \
       --orig_num "1000" --extra_num "0" \
       --J "10" \
       --modelVersion "Sphere_Kmeans_Pretrain" \
       --expName "demo"
   ```

3. **Run** SpherePair with KMeans clustering (pretrained):
   ```bash
   python run_model_Sphere_Kmeans.py \
       --dataset "fmnist" \
       --consRule "balance" \
       --consIndex "10" \
       --use_pretrain "True" \
       --epochs "300" \
       --expName "demo"
   # consIndex=10 => uses the 10th constraint set (10k constraints).
   # default --dim=10
   ```

### 4.2 Balanced Constraints + No Pretrain

You can skip `tool_pretrain_sphere.py` and change `modelVersion` accordingly:
```bash
# Generate constraints (modelVersion: Sphere_Kmeans_noPretrain)
python tool_createCons.py --dataset "fmnist" \
    --consRule "balance" \
    --set "train" \
    --orig_num "1000" --extra_num "9000" \
    --J "10" \
    --imbCluster "0" \
    --modelVersion "Sphere_Kmeans_noPretrain" \
    --expName "demo"

python tool_createCons.py --dataset "fmnist" \
    --consRule "balance" \
    --set "test" \
    --orig_num "1000" --extra_num "0" \
    --J "10" \
    --modelVersion "Sphere_Kmeans_noPretrain" \
    --expName "demo"

# Run (no pretrain)
python run_model_Sphere_Kmeans.py \
    --dataset "fmnist" \
    --consRule "balance" \
    --consIndex "10" \
    --use_pretrain "False" \
    --epochs "300" \
    --expName "demo"
```

### 4.3 Imbalanced Constraints + Pretrain

Similar process, but `consRule` is set to `extraCLs` for training set:
```bash
# 1) Pretrain (same as above)
python tool_pretrain_sphere.py --dataset "fmnist"

# 2) Generate imbalanced constraints
python tool_createCons.py --dataset "fmnist" \
    --consRule "extraCLs" \
    --set "train" \
    --orig_num "1000" --extra_num "9000" \
    --J "10" \
    --imbCluster "0" \
    --modelVersion "Sphere_Kmeans_Pretrain" \
    --expName "demo"

# For test set, usually keep it 'balance' for evaluation
python tool_createCons.py --dataset "fmnist" \
    --consRule "balance" \
    --set "test" \
    --orig_num "1000" --extra_num "0" \
    --J "10" \
    --modelVersion "Sphere_Kmeans_Pretrain" \
    --expName "demo"

# 3) Run
python run_model_Sphere_Kmeans.py \
    --dataset "fmnist" \
    --consRule "extraCLs" \
    --consIndex "10" \
    --use_pretrain "True" \
    --epochs "300" \
    --expName "demo"
# consIndex=10 => 1k balanced + 9k extra (i.e., IMB2 scale)
```

### 4.4 Imbalanced Constraints + No Pretrain

Just repeat the generation step with `modelVersion="Sphere_Kmeans_noPretrain"` and run `--use_pretrain "False"`.

---

## 5. Extended Experiments

You can test various models and datasets by looping over:

- **Datasets**: `mnist`, `fmnist`, `reuters`, `cifar10`, `stl10`, `imagenet10`, `cifar100`, etc.
- **Constraint Rules**: `balance` or `extraCLs`.
- **modelVersion**: for instance,
  - `Sphere_Kmeans_Pretrain` -> `run_model_Sphere_Kmeans.py --use_pretrain True`
  - `Sphere_Hierarchical_noPretrain` -> `run_model_Sphere_Hierarchical.py --use_pretrain False`
  - `CIDEC_Pretrain` -> `run_model_CIDEC.py --use_pretrain True`
  - `VanillaDCC` -> `run_model_VanillaDCC.py`, etc.

Make sure:

1. The same `--expName` in both `tool_createCons.py` and the `run_model_*.py` so that the correct constraint file path is loaded.
2. The same `--consRule` in both constraint generation and training scripts.
3. The `modelVersion` naming matches the `run_model_*.py` script you intend to run.
4. If the model needs a pretrained autoencoder, run `tool_pretrain_sdae.py` or `tool_pretrain_sphere.py` before training, with a matching `--dim`.

---

## 6. Hyperparameter Tuning (AutoEmbedder & VolMaxDCC)

For **AutoEmbedder** (tuning `alpha`) or **VolMaxDCC** (tuning `lambda`), you may need to run a grid search. Example outline:

- **AutoEmbedder**:
  1. Pretrain SDAE with `--finetune True`.
  2. Generate constraints (`balance`) and test multiple `alpha` values.
  3. Collect performance results and pick the best `alpha`.

- **VolMaxDCC**:
  1. Generate constraints (`balance`).
  2. Test multiple `lambda` values (e.g., `1e-1, 1e-2, ...`).
  3. Evaluate and choose the best.

You can find usage examples in the comments of `tool_createCons.py` or `run_model_*.py`.

---

## 7. Where to Find Logs/Results

All intermediate files, logs, models, and results will be stored under:
```
experiment/exp_{expName}/lab_{modelVersion}/{dataset}/{consRule}/
```
The exact subfolders/files may contain `cons.txt` (constraints), logs, checkpoints, etc., depending on the script.

---

## 8. FAQ

1. **Where are the generated constraint files stored?**  
   They are typically in `experiment/exp_{expName}/lab_{modelVersion}/{dataset}/{consRule}/cons_*.txt`, or similar naming patterns that include `consIndex`.

2. **How do I change the embedding dimension?**  
   Use `--dim <N>` in both the pretraining script and the corresponding run script. Make sure they match.

3. **Which script runs which modelVersion?**  
   See the mapping in Section [Extended Experiments](#extended-experiments). For example, `Sphere_Kmeans_Pretrain` corresponds to `run_model_Sphere_Kmeans.py --use_pretrain "True"`.

4. **Do I need to manually install packages?**  
   Once `environment.yml` is ready, you can just do `conda env create -f environment.yml`. Otherwise, install dependencies manually with `pip install ...`.

---

## 9. Citation

If you find this repository useful for your research, please cite our paper:

```
@inproceedings{your-spherepair-icml2025,
  title     = {SpherePair: An Anchor-Free Approach to Deep Constrained Clustering},
  author    = {Your Name and ...},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025},
}
```

---

## 10. License

This project is distributed under the [MIT License](./LICENSE) (if applicable).  
For other licensing or commercial inquiries, please contact the authors.
```
