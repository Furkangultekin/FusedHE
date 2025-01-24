import cv2
import numpy as np
from PIL import Image
import argparse
import torch
from models.model import FHE
from utils.inference_data import InferenceDataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser('inference', add_help=False)
    parser.add_argument('--folder_path', default='inference_data/', type=str) # learning rate
    parser.add_argument('--ckpt_path', default='path_to_ckpt', type=str)
    parser.add_argument('--output_dir', default='inference_results/', type=str)
    # [depth or full], depth: Only height head, full:height+Segmentation head
    parser.add_argument('--option', default='full', type=str) 
    parser.add_argument('--pre_trained', default=False, type=bool) 

    # [mit, conv, fused], Encoder type, mit: Hierarchical Vision transformers, conv: Resnet , Fused:mit+conv
    parser.add_argument('--enc_opt', default='fused', type=str) 
    parser.add_argument('--num_classes', default=1, type=int) #for segmentation head
    parser.add_argument('--in_channel_ffb', default=[2048, 1024, 512, 256], type=list)
                        #only enc_opt for conv, fused. usage = [2048, 1024, 512, 256]
    parser.add_argument('--out_channel_ffb', default=[512, 320, 128, 64], type=list)
                        #only enc_opt for mit, fused. usage = [512, 320, 128, 64] 
    parser.add_argument('--in_channel_decoderb', default=[1024, 640, 256], type=list)
                        #mit,fused = [1024, 640, 256] - conv = [1024, 512, 256]
    parser.add_argument('--out_channel_decoderb', default=[640, 256, 128], type=list)
                        #mit,fused = [640, 256, 128] - conv = [512, 256, 128]
    parser.add_argument('--out_decoder', default=64, type=int)
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--max_height_eval', default=185, type=int)
    parser.add_argument('--min_height_eval', default=1e-4, type=float)

    return parser

def process_small_or_exact_image(image, model, window_size=(512, 512)):
    """
    Handles inference for images smaller than or exactly 512x512 by padding or direct prediction.
    :param image: Input tensor of shape (batch_size, channels, height, width)
    :param model: Model for prediction
    :param h: Height of the image
    :param w: Width of the image
    :param window_size: Expected window size (default: 512x512)
    :return: Model prediction output
    """
    b,c,h,w = image.shape
    if h < window_size[0] or w < window_size[1]:
        # Pad image to 512x512
        padded_image = torch.zeros((image.size(0), image.size(1), window_size[0], window_size[1]), device=image.device)
        padded_image[:, :, :h, :w] = image
        with torch.no_grad():
            pred = model(padded_image)

        prediction = pred['pred_d'].squeeze()
        return prediction[:h, :w]  # Crop back to original size

    # For exactly 512x512
    with torch.no_grad():
        pred = model(image)
        return pred['pred_d'].squeeze()

def sliding_window_inference(image, model, window_size=(512, 512), stride=312):
    """
    Performs inference using the sliding window method and averages overlapping regions.
    Handles images larger than 512x512.
    :param image: Input tensor of shape (batch_size, channels, height, width)
    :param model: Model for prediction
    :param h: Height of the image
    :param w: Width of the image
    :param window_size: Window size (e.g., (512, 512))
    :param stride: Sliding step size (e.g., 312 pixels)
    :return: Combined prediction output (in the original image size)
    """
    b, c, h, w = image.shape

    # Pad the image to ensure edge information is preserved
    pad_h = (stride - (h - window_size[1]) % stride) % stride
    pad_w = (stride - (w - window_size[0]) % stride) % stride
    padded_image = torch.zeros((b, c, h + pad_h, w + pad_w), device=image.device)
    padded_image[:, :, :h, :w] = image

    new_h, new_w = padded_image.shape[2], padded_image.shape[3]

    # Initialize output and weight matrices
    output = torch.zeros((b, new_h, new_w), device=image.device, dtype=torch.float32)
    weight_matrix = torch.zeros((b, new_h, new_w), device=image.device, dtype=torch.float32)

    # Perform sliding window inference for large images
    total_steps = ((new_h - window_size[1]) // stride + 1) * ((new_w - window_size[0]) // stride + 1)
    step = 0

    with tqdm(total=total_steps, desc="Sliding Window Inference", unit="window") as pbar:
        for y in range(0, new_h - window_size[1] + 1, stride):
            for x in range(0, new_w - window_size[0] + 1, stride):
                # Crop the window
                window = padded_image[:, :, y:y + window_size[1], x:x + window_size[0]]

                # Get prediction from the model
                with torch.no_grad():
                    pred = model(window)

                prediction = pred['pred_d'].squeeze(1)  # Shape: (batch_size, height, width)
                # Combine the output and weights
                output[:, y:y + window_size[1], x:x + window_size[0]] += prediction
                weight_matrix[:, y:y + window_size[1], x:x + window_size[0]] += 1

                step += 1
                pbar.update(1)

    # Normalize the output using the weight matrix
    output = torch.div(output, weight_matrix)

    # Remove the padding
    output = output[:, :h, :w]

    return output.squeeze()

def main(args):
    device = torch.device(args.device)
    print('Loading the model...')
    model = FHE(args)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    print('Model is Loaded.')
    inference_data = InferenceDataset(args.folder_path, args=args)
    inference_loader = DataLoader(inference_data, batch_size=1, shuffle=False)
    print('Processing is starting on given inference data...')
    for batch_idx, batch in enumerate(inference_loader):      
        input_RGB = batch['image'].to(device)
        input_name = batch['fname'][0]
        print('Processing: ', input_name)
        print(input_RGB.shape)
        b, c, h, w = input_RGB.shape
        if h <= 512 and w <= 512:
            output = process_small_or_exact_image(input_RGB, model, window_size=(512, 512))
        else:
            output = sliding_window_inference(input_RGB, model, window_size=(512, 512), stride=312)

        output_image_path = os.path.join(args.output_dir, f"height_{input_name}")
        output_np = ((output.cpu().numpy())*args.max_height_eval).astype(np.float32)
        cv2.imwrite(output_image_path, output_np)
        print(output_image_path, ' is done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser('FusedHE inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)
