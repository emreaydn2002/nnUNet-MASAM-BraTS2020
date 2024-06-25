
# MASAM-BraTS2020

This repository contains the implementation of nnUNet with MASAM (Multi-Scale Attention Mechanism) for brain tumor segmentation using the BraTS2020 dataset.

## Setup and Installation

To run this code on Google Colab, follow these steps:

### 1. Create a New Notebook and Mount Google Drive

Create a new notebook in Google Colab and mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Download the Dataset

Download the BraTS2020 dataset from [this link](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) and upload the zip file to a location in your Google Drive. Set the dataset and zip paths accordingly:

```python
dataset_directory = '/content/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

# Example usage
zip_path = '/content/drive/MyDrive/dataset_brats/archive.zip'
```

### 3. Install Required Packages

Install the necessary packages using the `requirements.txt` file. Ensure that the `requirements.txt` file is located in your Google Drive or Colab environment (you can find the file in the repository):

```python
!pip install -r /content/drive/MyDrive/dataset_brats/requirements.txt
```

After installing the packages, restart the runtime.

### 4. Download vit_b File

Download the vit_b file from [this link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and upload it to your Google Drive or Colab environment. Set the `ckpt` variable accordingly:

```python
ckpt = '/content/drive/MyDrive/dataset_brats/sam_vit_b_01ec64.pth'
```

### 5. Define Training and Configuration Parameters

Define the training and configuration parameters:

```python
args = {
    'root_path': '/content/data',  # Root directory of the dataset
    'output': '/content/output',  # Directory to save output files
    'num_classes': 1+3,  # Number of classes (1+3 = 4)
    'batch_size': 24,  # Mini-batch size
    'n_gpu': 1,  # Number of GPUs to use
    'base_lr': 0.0001,  # Learning rate
    'max_epochs': 20,  # Maximum number of epochs
    'stop_epoch': 20,  # Epoch to stop training
    'deterministic': 1,  # Deterministic mode for reproducibility
    'img_size': 240,  # Image size
    'seed': 1234,  # Seed for random number generator
    'vit_name': 'vit_b',  # Model name
    'ckpt': '/content/drive/MyDrive/dataset_brats/sam_vit_b_01ec64.pth',  # Checkpoint file path
    'adapt_ckpt': None,  # Adaptation checkpoint file path
    'rank': 32,  # Rank value
    'scale': 1.0,  # Scale value
    'warmup': True,  # Use warmup
    'warmup_period': 100,  # Warmup period
    'AdamW': True,  # Use AdamW optimization
    'module': 'sam_fact_tt_image_encoder',  # Module name
    'dice_param': 0.8,  # Dice parameter
    'lr_exp': 7,  # Learning rate exponent value
    'tf32': True,  # Use TF32
    'compile': False,  # Whether to compile the model
    'use_amp': True,  # Use automatic mixed precision
    'skip_hard': True  # Whether to skip hard examples
}
```

Make any necessary changes to the configuration settings if needed.

### 6. Download the MASAM Code

Download the MASAM code from [this link](https://github.com/cchen-cc/MA-SAM) and upload the entire code to a location in your Google Drive. Replace the `test.py` file in the repository with the `test.py` file provided.

### 7. Run Tests

Run the test script with the following command (adjust paths as necessary):

```python
!python /content/drive/MyDrive/dataset_brats/MA-SAM-main/MA-SAM/test.py --adapt_ckpt /content/output/epoch_19.pth --data_path /content/data --vit_name vit_b --ckpt /content/drive/MyDrive/dataset_brats/sam_vit_b_01ec64.pth --is_savenii
```

In this command:
- `output/epoch_19.pth` represents the most recent model checkpoint. Adjust the path based on your latest epoch.
- `sam_vit_b_01ec64.pth` should be set to the path of your uploaded vit_b file.

## Project Structure

- `nnUNet-MASAM-BraTS2020/`
  - `503020220054_EmreAydin_masam.ipynb`: The main notebook containing the implementation of nnUNet with MASAM for brain tumor segmentation.
  - `data/`: Directory to store the BraTS2020 dataset.
  - `results/`: Directory to save the model outputs and evaluation results.
  - `requirements.txt`: File containing the list of required packages.

## References

- nnUNet: [GitHub Link](https://github.com/MIC-DKFZ/nnUNet)
- BraTS2020 Dataset: [BraTS2020 Data](https://www.med.upenn.edu/cbica/brats2020/data.html)
- MASAM: [GitHub Link](https://github.com/cchen-cc/MA-SAM)

## Acknowledgements

This work is based on the nnUNet framework and incorporates the MASAM approach for improved segmentation performance.
