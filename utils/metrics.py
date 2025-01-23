import torch
import numpy as np

def eval_depth(pred, target):
    assert pred.shape == target.shape
    pred[pred==0]=0.0001
    target[target<=0.0001]=0.0001
    mask = target>=2
    pred_mask = pred[mask]
    target_mask = target[mask]
    thresh = torch.max((pred / target), (target / pred))

    volume = torch.min((torch.sum(pred) / torch.sum(target)), (torch.sum(target) / torch.sum(pred)))

    d1 = torch.sum(thresh < 1.25).float() / (len(thresh)*len(thresh))
    d2 = torch.sum(thresh < 1.25 ** 2).float() / (len(thresh)*len(thresh))
    d3 = torch.sum(thresh < 1.25 ** 3).float() / (len(thresh)*len(thresh))

    diff_mask = pred_mask - target_mask
    diff_log_mask = torch.log(pred_mask) - torch.log(target_mask)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel_mask = torch.mean(torch.abs(diff_mask) / target_mask)
    sq_rel_mask = torch.mean(torch.pow(diff_mask, 2) / target_mask)

    rmse= torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    rmse_mask = torch.sqrt(torch.mean(torch.pow(diff_mask, 2)))
    rmse_log_mask = torch.sqrt(torch.mean(torch.pow(diff_log_mask , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    binary = pred>=2
    segment = mask
    binary = binary.float()
    segment = segment.float() 
    intersection = (binary + segment) == 2
    union = (binary + segment) >=1
    iou = (torch.sum(intersection)) / (torch.sum(union) + 1)

    target_mask[target_mask<3]=3
    pred_mask[pred_mask<3]=3
    diff_mask_3m = pred_mask-target_mask
    abs_rel_mask_3m = torch.mean(torch.abs(diff_mask_3m) / target_mask)
    sq_rel_mask_3m = torch.mean(torch.pow(diff_mask_3m, 2) / target_mask)

    target_mask[target_mask<5]=5
    pred_mask[pred_mask<5]=5
    diff_mask_5m = pred_mask-target_mask
    abs_rel_mask_5m = torch.mean(torch.abs(diff_mask_5m) / target_mask)
    sq_rel_mask_5m = torch.mean(torch.pow(diff_mask_5m, 2) / target_mask)

    return {'d1': d1, 'd2': d2, 'd3': d3, 'abs_rel_mask': abs_rel_mask,'abs_rel_mask_3m': abs_rel_mask_3m,'abs_rel_mask_5m':abs_rel_mask_5m,
            'sq_rel_mask': sq_rel_mask,'sq_rel_mask_3m':sq_rel_mask_3m,'sq_rel_mask_5m':sq_rel_mask_5m, 'rmse': rmse, 'rmse_mask':rmse_mask, 'segment_iou': iou, 
            'rmse_log': rmse_log,'rmse_log_mask':rmse_log_mask, 'log10':log10, 'silog':silog, 'volume':volume}
