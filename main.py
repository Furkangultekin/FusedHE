import argparse
import datetime
import random
import os
from pathlib import Path

import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader, random_split

from torch import nn, optim
import utils.misc as utils
from utils.local_dataset import LocalDataset
from utils.criterion import DepthSegLoss
from models.model import FHE
from train import train,validate
import copy

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def get_args_parser():
    parser = argparse.ArgumentParser('FusedHE', add_help=False)
    parser.add_argument('--lr', default=7e-5, type=float) # learning rate
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int) #batch size
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=70, type=int) # epochs
    parser.add_argument('--lr_drop', default=[400], type=list)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--lambd', default=0,type=float) # coeffitiont of Segmentation Loss
    parser.add_argument('--sgd', action='store_true')

    # [depth or full], depth: Only height head, full:height+Segmentation head
    parser.add_argument('--option', default='full', type=str) 
    # [mit, conv, fused], Encoder type, mit: Hierarchical Vision transformers, conv: Resnet , Fused:mit+conv
    parser.add_argument('--enc_opt', default='fused', type=str) 
    parser.add_argument('--pre_trained', default=True, type=bool)
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
    parser.add_argument('--job_name', default='fused_with_segment_head', type=str)

    # dataset parameters
    parser.add_argument('--dataset_name', default='DFC2023') #voxdata_coco  scenecad stru3d
    parser.add_argument('--train_dir', default='path_to_train_data_folder', type=str) #dfc2023/train/
    parser.add_argument('--val_dir', default='path_to_validation_data_folder', type=str) #dfc2023/validation/

    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--val_freq', default=1, type=int)
    parser.add_argument('--exp_name', default=True, type=bool)
    parser.add_argument('--save_model', default=True, type=bool)
    parser.add_argument('--save_results', default=True, type=bool)
    parser.add_argument('--max_depth_eval', default=185, type=int)
    parser.add_argument('--min_depth_eval', default=1e-4, type=float)
    parser.add_argument('--dataset', default="local", type=bool)
    
    return parser

def main(args):
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    # setup wandb for logging
    utils.setup_wandb()
    wandb.init(project="fused-he")
    wandb.run.name = args.run_name

    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dir = args.train_dir
    val_dir = args.val_dir

    train_data = LocalDataset(train_dir)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,shuffle=True)

    val_data = LocalDataset(val_dir)
    val_loader = DataLoader(val_data, batch_size=args.batch_size,shuffle=True)

    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }
    dataset_sizes = {
        "train": len(train_data),
        "val": len(val_data)
    }
    model = FHE(args)
    model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    criterion_d = DepthSegLoss(args=args)
    #criterion_s = nn.CrossEntropyLoss()
    optimizer_d = optim.Adam(model.parameters(), args.lr)

    schedular = optim.lr_scheduler.StepLR(optimizer_d)

    model.to(device)

    

    best_acc = 100.0
    best_model_wts = copy.deepcopy(model.state_dict())
    all_loss_tra = []
    all_loss_val = []
    for epoch in range(1, args.epochs + 1):
        print('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        total_loss, depth_loss,segment_loss,model = train(train_loader, model, criterion_d, optimizer_d=optimizer_d,
                                scheduler = schedular, 
                            device=device, epoch=epoch, args=args,option=args.option)
        train_log_stats = { "train/total_loss": total_loss,
                            "train/depth_loss": depth_loss,
                           "train/segment_loss" : segment_loss

        }

        wandb.log({"epoch": epoch})
        wandb.log({"lr_rate": args.lr})
        wandb.log(train_log_stats)
        print(train_log_stats)

        if epoch % args.val_freq == 0:
            results_dict, total_loss_val, depth_loss_val,loss_s_val, model = validate(val_loader, model, criterion_d, 
                                                device=device, epoch=epoch, args=args,
                                                log_dir=args.output_dir,option=args.option)

            val_log_stats = {"val/total_loss": total_loss_val,
                            "val/depth_loss": depth_loss_val,
                            "val/segment_loss": loss_s_val,
                            "val/d1": results_dict['d1'],
                            "val/d2": results_dict['d2'],
                            "val/d3": results_dict['d3'],
                            "val/abs_rel_mask": results_dict['abs_rel_mask'],
                            "val/sq_rel_mask": results_dict['sq_rel_mask'],
                            "val/rmse": results_dict['rmse'],
                            "val/segment_iou": results_dict['segment_iou'],
                            "val/rmse_log": results_dict['rmse_log'],
                            "val/log10": results_dict['log10'],
                            "val/silog": results_dict['silog'],
                             }

            print(val_log_stats)
            wandb.log(val_log_stats)

            epoch_rmse = results_dict["rmse"]
            if epoch_rmse < best_acc:
                print('best')
                best_acc = epoch_rmse
                best_model_wts = copy.deepcopy(model.state_dict()) # keep the best validation accuracy model
        

            with open(f"{args.output_dir}/log.txt", 'a') as txtfile:
                txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
                txtfile.write(f'{train_log_stats}')     
                txtfile.write(f'{val_log_stats}')

    torch.save(best_model_wts, os.path.join(args.output_dir,'best.ckpt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RoomFormer training script', parents=[get_args_parser()])
    args = parser.parse_args()
    now = datetime.datetime.now()
    run_id = now.strftime("%Y-%m-%d-%H-%M-%S")
    args.run_name = run_id+'_'+args.job_name 
    args.output_dir = os.path.join(args.output_dir, args.run_name)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)






