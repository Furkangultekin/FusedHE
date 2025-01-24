# FusedHE

**Fused-HE: Fused Convolution and Vision Transformers Encoder for Height Estimation from Monoscopic Satellite Images**

## Overview

FusedHE is a deep learning model designed to estimate height information from single (monoscopic) satellite images. By integrating convolutional neural networks (CNNs) with Vision Transformer (ViT) encoders, FusedHE aims to enhance the accuracy of height estimations in remote sensing applications.

## Features

- **Hybrid Architecture**: Combines the strengths of CNNs and Vision Transformers to capture both local and global features from satellite imagery.
- **Monoscopic Input**: Processes single-view satellite images, eliminating the need for stereoscopic data.
- **Height Estimation**: Outputs precise height predictions, beneficial for various geospatial analyses.

## Installation

1. **Clone the Repository**:
  ```bash
  git clone https://github.com/Furkangultekin/FusedHE.git
  cd FusedHE
  ```
2. **Set Up the Environment**:
  - It's recommended to use a virtual environment to manage dependencies.
  - Install the required packages:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

1. **Prepare your data**:
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
    







