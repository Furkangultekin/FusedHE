import os
import torch
import utils.metrics as metrics
import utils.logging as logging

metric_name = ['d1', 'd2', 'd3', 'abs_rel_mask', 'abs_rel_mask_3m','abs_rel_mask_5m', 'sq_rel_mask','sq_rel_mask_3m' ,'sq_rel_mask_5m', 'rmse', 'rmse_mask','segment_iou', 'rmse_log','rmse_log_mask',
               'log10', 'silog','volume']

global global_step
global_step = 0

def train(train_loader, model, criterion_d, optimizer_d ,scheduler, device, epoch, args, option="full"):    
    
    global global_step

    model.train()
    depth_loss = logging.AverageMeter()
    segment_loss =  logging.AverageMeter()
    total_loss = logging.AverageMeter()

    for batch_idx, batch in enumerate(train_loader):      
        global_step += 1

        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        segment_gt = batch['segment'].to(device) 

        preds = model(input_RGB)

        depth_gt = depth_gt.squeeze()
        depth_seg = segment_gt.squeeze()  
        loss, loss_d, loss_s = criterion_d(preds,depth_gt,depth_seg)        

        optimizer_d.zero_grad()

        depth_loss.update(loss_d.item(), input_RGB.size(0))
        total_loss.update(loss.item(), input_RGB.size(0))
        segment_loss.update(loss_s.item(), input_RGB.size(0))
        loss.backward()
        optimizer_d.step()

        logging.progress_bar(batch_idx, len(train_loader), args.epochs, epoch,
                            ('Depth Loss: %.4f (%.4f)' %
                            (total_loss.val, total_loss.avg)))
        
        scheduler.step()

    return total_loss.avg, depth_loss.avg, segment_loss.avg, model


def validate(val_loader, model, criterion_d, device, epoch, args, log_dir, option="full"):
    depth_loss = logging.AverageMeter()
    segment_loss = logging.AverageMeter()
    total_loss = logging.AverageMeter()
    model.eval()

    if args.save_model and epoch>=65 :
        torch.save(model.state_dict(), os.path.join(
            log_dir, 'epoch_%02d_model.ckpt' % epoch))

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0
    a=0
    for batch_idx, batch in enumerate(val_loader):
        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        segment_gt = batch['segment'].to(device) 
        #filename = batch['filename'][0]

        with torch.no_grad():
            preds = model(input_RGB)

        depth_gt = depth_gt.squeeze()
        depth_seg = segment_gt.squeeze()
        loss, loss_d, loss_s = criterion_d(preds,depth_gt,depth_seg)

        depth_loss.update(loss_d.item(), input_RGB.size(0))

        total_loss.update(loss.item(), input_RGB.size(0))
        segment_loss.update(loss_s.item(), input_RGB.size(0))
        computed_result = metrics.eval_depth(preds['pred_d'].squeeze(), depth_gt)

        logging.progress_bar(batch_idx, len(val_loader), args.epochs, epoch)

        for key in result_metrics.keys():
            result_metrics[key] += computed_result[key]

        a+=1

    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (a + 1)
    
        


    return result_metrics, total_loss.avg, depth_loss.avg, segment_loss.avg, model



