# FusedHE

**Fused-HE: Fused Convolution and Vision Transformers Encoder for Height Estimation from Monoscopic Satellite Images**

## Overview

FusedHE is a deep learning model designed to estimate height information from single (monoscopic) satellite images. By integrating convolutional neural networks (CNNs) with Vision Transformer (ViT) encoders, FusedHE aims to enhance the accuracy of height estimations in remote sensing applications.

![3d_barca](https://github.com/user-attachments/assets/124da4c4-8931-4783-a90c-e812d13e2939)
![3d_2](https://github.com/user-attachments/assets/3633da4e-32c3-4854-a0d2-fffd5caa4355)

## Features

- **Hybrid Architecture**: Combines the strengths of CNNs and Vision Transformers to capture both local and global features from satellite imagery.
- **Monoscopic Input**: Processes single-view satellite images, eliminating the need for stereoscopic data.
- **Height Estimation**: Outputs precise height predictions, beneficial for various geospatial analyses.

## MODELS:
  The following models can be used for training with this repository:
  - **CNN**: Convolutional Neural Network height estimation algorithm using [ResNet-101](https://arxiv.org/abs/1512.03385) encoder.
  - **MiT**: Hierarchical Vision Transformers from [SegFormer](https://arxiv.org/abs/2105.15203) paper for height estimation using MiT (Mixed Transformers) encoder.
  - **[Fused-HE](https://open.metu.edu.tr/handle/11511/108758)**: Encoder created by the fusion of convolutional and vision transformer encoders.
  - **[FusedSeg-HE](https://open.metu.edu.tr/handle/11511/108758)**: Fused encoder with additional segment head to increase the accuracy.

## Installation

### 1. Clone the Repository:
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

    Dowload link: https://drive.google.com/drive/folders/1EwmFM0zHENQGPiMemXfe9W97qUKsyLhG?usp=drive_link
  
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

## Important Parameters
### Training:

  Check the following arguments on `main.py`:
  -  `--lr`: learning rate
  -  `--batch_size`: batch size
  -  `--epochs` : epochs
  -  `--lamd` : Coefficient for segmentation loss in Total loss of the FusedSeg-HE model. `lamd=0` segmentation head will not effect on estimated height. Recommended value `lamd=0.005`
  -  `--option` : [depth or full], Head option to define model. `option=depth` --> The model use only height head. `option=full` --> The model use height head and segment head together.
  -  `--enc_opt` : [mit, conv, fused], Defining encoder type, `enc_opt=conv` --> ResNet-101 encoder, `enc_opt=mit` --> MiT encoder, `enc_opt=fused` --> ResNet-101 + MiT
  -  `--pre_trained` : [True, False], encoder pre-trained parameters
  -  `--num_classes` : for segmentation head
  -  `--in_channel_decoderb` : list of decoder blocks input size. `enc_opt=mit` or `enc_opt=fused` --> [1024, 640, 256] , `enc_opt=conv` --> [1024, 512, 256].
  -  `--out_channel_decoderb` : list of decoder blocks output size. `enc_opt=mit` or `enc_opt=fused` --> [640, 256, 128] , `enc_opt=conv` --> [512, 256, 128].
  -  `--train_dir` : Training dataset directory path
  -  `--val_dir` : Validation dataset directory path
  -  `--mit_ckpt_path` : `mit_b4.pth` parameters file path
  -  `--max_height_eval` : Maximum height value that ground truth nDSM data contain.
  -  `--min_height_eval` : Minimum height value that gorund truth nDSM data contain.

### Inference:
  Check the following arguments on `inference.py`:
  - `--folder_path` : inference data folder path.
  - `--ckpt_path` : Path to .ckpt ex. FusedSegHE.ckpt
  - `--output_dir` : Inference data result dir.
    







