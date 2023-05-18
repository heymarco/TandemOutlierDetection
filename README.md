# Tandem Outlier Detectors for Decentralized Data

Code for the short paper *Tandem Outlier Detectors for Decentralized Data* published at SSDBM22.

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
