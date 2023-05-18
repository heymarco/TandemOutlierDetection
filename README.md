# Tandem Outlier Detectors for Decentralized Data

Code for the short paper *Tandem Outlier Detectors for Decentralized Data* published at SSDBM22.

https://dl.acm.org/doi/10.1145/3538712.3538748

## Abstract

Today, the collection of decentralized data is a common scenario: smartphones store users' messages locally, smart meters collect energy consumption data, and modern power tools monitor operator behavior. We identify different types of outliers in such data: local, global, and partition outliers}. They contain valuable information, for example, about mistakes in operation. However, existing outlier detection approaches cannot distinguish between those types. Thus, we propose a "tandem" technique to join "local" and "federated" outlier detectors. Our core idea is to combine outlier detection on a single device with latent information about devices' data to discriminate between different outlier types. To the best of our knowledge, our method is the first to achieve this.
We evaluate our approach on publicly available synthetic and real-world data that we collect in a study with 15 participants operating power tools.

## Dependencies

`conda env create -f environment.yml` will create a conda environment named `TandemOutlierDetection` with all necessary dependencies installed.
- use `conda activate TandemOutlierDetection` to activate the environment.

If automatic installation fails, you can also install the dependencies by hand:
- Python 3.8
- **Conda:** `pandas, numpy, torch, keras, seaborn, tensorflow, torchvision`
- **Pip** `flwr, tqdm, tsfel`

## Running experiments

Execute the following commands to reproduce the experiments from the paper. All results will be stored as csv-files `results/result.csv`. (Be sure to rename or remove the old file before you run a new experiment!)

Synthetic data with local / global outliers:
- `bash bash/start_experiment.sh -t local/global -r 20 -R 60`

Synthetic data with partition outliers
- `bash bash/start_experiment.sh -t partition_outlier -r 20 -R 150`

Power tool data from the study
- You can download the data from this link: https://drive.google.com/drive/folders/1A88h3p-1LY_H2JWZMAU9Cmg7iSaqHUEw?usp=sharing
- `bash bash/start_experiment.sh -t powertool -r 20 -R 10`
