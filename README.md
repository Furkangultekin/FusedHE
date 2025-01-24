# FusedHE

**Fused-HE: Fused Convolution and Vision Transformers Encoder for Height Estimation from Monoscopic Satellite Images**

## Overview

FusedHE is a deep learning model designed to estimate height information from single (monoscopic) satellite images. By integrating convolutional neural networks (CNNs) with Vision Transformer (ViT) encoders, FusedHE aims to enhance the accuracy of height estimations in remote sensing applications.

## Features

- **Hybrid Architecture**: Combines the strengths of CNNs and Vision Transformers to capture both local and global features from satellite imagery.
- **Monoscopic Input**: Processes single-view satellite images, eliminating the need for stereoscopic data.
- **Height Estimation**: Outputs precise height predictions, beneficial for various geospatial analyses.

## Installation

### 1.Clone the Repository:
  ```bash
  git clone https://github.com/Furkangultekin/FusedHE.git
  cd FusedHE
  ```
### 2. Set Up the Environment:
  - It's recommended to use a virtual environment to manage dependencies using Python 3.10.11.
  - Install the required packages:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

### 1. Prepare your data:
  - Ensure your 512x512 monoscopic satellite images and ground truths are organized in a directory.
  - Example dataset structure:
    ```bash
    .
    ├── dfc2023/                             # dataset main folder
         ├── train/                          # training dataset folder
         │    ├── rgb                        # input image folder
         │    │    ├── image_1.tif          
         │    │    ├── image_2.tif
         │    │    └── ....
         │    ├── dsm                        #ground truth nDSM folder
         │    │    ├── image_1.tif
         │    │    ├── image_2.tif
         │    │    └── ....
         └── validation/                     #validation data folder
              ├── rgb
              │    ├── image_1.tif
              │    ├── image_2.tif
              │    └── ....
              └── dsm
                   ├── image_1.tif
                   ├── image_2.tif
                   └── ....          

    ```
    > **Note that**: Currently input folder name must be 'rgb' and and ground truth folder name must be 'dsm'.
    > 
    > **Naming Convention**: Each input image in the `rgb/` folder must have a corresponding ground truth file in the `dsm/` folder with the same name (e.g., `image_001.tif` in `rgb/` matches `image_001.tif` in `dsm/`).

### 2. Download the checkpoints:
   
    You will need to download the following model checkpoints to run the project:
  - **MIT Pretrained Parameters**: `mit_b4.pth` (used for the Vision Transformer encoder, can be used during training)
  - **Trained Parameters on DataFusion Contest 2023 Dataset**: `FusedSeg-HE.ckpt` (FusedSeg-HE parameters, can be used for inference data)

    Dowload link: https://drive.google.com/file/d/1FeX67612TZtzazPOJzqjkT-nweoJ2p-f/view?usp=drive_link
  
### 3. Training the model:
  - To train FusedSeg-HE on your dataset, run:
    
  ```bash
  python main.py --train_dir /path/to/train/folder --val_dir /path/to/validation/folder --mit_ckpt_path /path/to/mit_b4.pth
  ```

  > Adjust the parameters as needed.

### 4. Inference:
   For height estimation on new images:
  ```bash
  python .\inference.py --ckpt_path path/to/FusedSegHE.ckpt
  ```

## Parameters for training






