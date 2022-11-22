import argparse
import json
import os
import os.path as osp
import random
import sys
import time
from imgaug import augmenters as iaa
import numpy as np
import torch
from mmcv import Config
import torchvision.transforms as transforms
from PIL import Image
from dataset import build_data_loader
from models import build_model
from utils import AverageMeter
import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(9797)
torch.cuda.manual_seed(9797)
np.random.seed(9797)
random.seed(9797)
EPS = 1e-6
from torch.utils.tensorboard import SummaryWriter



def init_tensorboard(out_dir: str = 'logs'):
    if not os.path.exists(out_dir):  ##目录存在，返回为真
        os.makedirs(out_dir)

    writer = SummaryWriter(log_dir=out_dir)
    ''' 
    https://pytorch.org/docs/stable/tensorboard.html
    writer.
    add_scalar(tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False)
    add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None)
    add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
    add_images(tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW')
    '''

    #  writer.close()  需在最后关闭
    return writer
def eval(train_loader, model,  epoch,  cfg):
    with torch.no_grad():
        model.train()

        # meters
        batch_time = AverageMeter(max_len=500)
        data_time = AverageMeter(max_len=500)
        losses = AverageMeter(max_len=500)
        losses_text = AverageMeter(max_len=500)
        losses_kernels = AverageMeter(max_len=500)
        losses_emb = AverageMeter(max_len=500)
        ious_text = AverageMeter(max_len=500)
        ious_kernel = AverageMeter(max_len=500)

        # start time
        start = time.time()
        for iter, data in enumerate(train_loader):
            # skip previous iterations

            # time cost of data loader
            data_time.update(time.time() - start)

            # prepare input
            data.update(dict(cfg=cfg))

            # forward
            outputs = model(**data)
            # *************************************************************************
            # print(outputs)

            # detection loss
            loss_text = torch.mean(outputs['loss_text'])
            losses_text.update(loss_text.item(), data['imgs'].size(0))

            loss_kernels = torch.mean(outputs['loss_kernels'])
            losses_kernels.update(loss_kernels.item(), data['imgs'].size(0))
            if 'loss_emb' in outputs.keys():
                loss_emb = torch.mean(outputs['loss_emb'])
                losses_emb.update(loss_emb.item(), data['imgs'].size(0))
                loss = loss_text + loss_kernels + loss_emb
            else:
                loss = loss_text + loss_kernels

            iou_text = torch.mean(outputs['iou_text'])
            ious_text.update(iou_text.item(), data['imgs'].size(0))
            iou_kernel = torch.mean(outputs['iou_kernel'])
            ious_kernel.update(iou_kernel.item(), data['imgs'].size(0))

            losses.update(loss.item(), data['imgs'].size(0))

            batch_time.update(time.time() - start)

            # update start time
            start = time.time()

        

    # print log
    if True:
        writer.add_scalar('Val-Loss', losses.avg, global_step=epoch)
        writer.add_scalar('Val-Loss-text', losses_text.avg, global_step=epoch)
        writer.add_scalar('Val-Loss-kernel', losses_kernels.avg, global_step=epoch)
        writer.add_scalar('Val-Loss-emb', losses_emb.avg, global_step=epoch)
        # writer.add_scalar('Loss-rec', losses_rec.avg, global_step=epoch)

        writer.add_scalar('Val-IoU(text)', ious_text.avg, global_step=epoch)
        writer.add_scalar('Val-IoU(kernel)', ious_kernel.avg, global_step=epoch)
        # writer.add_scalar('ACC rec', accs_rec.avg, global_step=epoch)

        log = f'EVALUATION-LOG' \
                f'Total-Time: {batch_time.avg * iter / 60.0:.0f}min | ' \
                f'Loss: {losses.avg:.3f} | ' \
                f'IoU(text/kernel): {ious_text.avg:.3f}/{ious_kernel.avg:.3f}' #\
                
                # f'{" | ACC rec: " + format(accs_rec.avg, ".3f") if with_rec else ""}'
        print(log, flush=True)



gs = 0 # global step
def train(train_loader, model, optimizer, epoch, start_iter, cfg):
    model.train()

    # meters
    batch_time = AverageMeter(max_len=500)
    data_time = AverageMeter(max_len=500)

    losses = AverageMeter(max_len=500)
    losses_text = AverageMeter(max_len=500)
    losses_kernels = AverageMeter(max_len=500)
    losses_emb = AverageMeter(max_len=500)
    losses_rec = AverageMeter(max_len=500)

    ious_text = AverageMeter(max_len=500)
    ious_kernel = AverageMeter(max_len=500)
    accs_rec = AverageMeter(max_len=500)

    with_rec = hasattr(cfg.model, 'recognition_head')
    
    # start time
    start = time.time()
    for iter, data in enumerate(train_loader):
        # skip previous iterations
        if iter < start_iter:
            print('Skipping iter: %d' % iter)
            continue

        # time cost of data loader
        data_time.update(time.time() - start)

        # adjust learning rate
        adjust_learning_rate(optimizer, train_loader, epoch, iter, cfg)

        # prepare input
        data.update(dict(cfg=cfg))

        
        outputs = model(**data)
        # *************************************************************************
        # print(outputs)
        # print(r'--------------------------------------------------------------------')
        # detection loss
        loss_text = torch.mean(outputs['loss_text'])
        losses_text.update(loss_text.item(), data['imgs'].size(0))

        loss_kernels = torch.mean(outputs['loss_kernels'])
        losses_kernels.update(loss_kernels.item(), data['imgs'].size(0))
        if 'loss_emb' in outputs.keys():
            loss_emb = torch.mean(outputs['loss_emb'])
            losses_emb.update(loss_emb.item(), data['imgs'].size(0))
            loss = loss_text + loss_kernels + loss_emb
        else:
            loss = loss_text + loss_kernels

        iou_text = torch.mean(outputs['iou_text'])
        ious_text.update(iou_text.item(), data['imgs'].size(0))
        iou_kernel = torch.mean(outputs['iou_kernel'])
        ious_kernel.update(iou_kernel.item(), data['imgs'].size(0))

        # recognition loss
        if with_rec:
            loss_rec = outputs['loss_rec']
            valid = loss_rec > -EPS
            if torch.sum(valid) > 0:
                loss_rec = torch.mean(loss_rec[valid])
                losses_rec.update(loss_rec.item(), data['imgs'].size(0))
                loss = loss + loss_rec

                acc_rec = outputs['acc_rec']
                acc_rec = torch.mean(acc_rec[valid])
                accs_rec.update(acc_rec.item(), torch.sum(valid).item())

        # if cfg.debug:
        #     from IPython import embed
        #     embed()

        losses.update(loss.item(), data['imgs'].size(0))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)

        # update start time
        start = time.time()

        
        if iter % 20 == 0:
            length = len(train_loader)
            log = f'({iter + 1}/{length}) ' \
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f} | ' \
                  f'Batch: {batch_time.avg:.3f}s | ' \
                  f'Total: {batch_time.avg * iter / 60.0:.0f}min | ' \
                  f'ETA: {batch_time.avg * (length - iter) / 60.0:.0f}min | ' \
                  f'Loss: {losses.avg:.3f} | ' \
                  f'Loss(text/kernel/emb{"/rec" if with_rec else ""}): ' \
                  f'{losses_text.avg:.3f}/{losses_kernels.avg:.3f}/' \
                  f'{losses_emb.avg:.3f}' \
                  f'{"/" + format(losses_rec.avg, ".3f") if with_rec else ""} | ' \
                  f'IoU(text/kernel): {ious_text.avg:.3f}/{ious_kernel.avg:.3f}' \
                  f'{" | ACC rec: " + format(accs_rec.avg, ".3f") if with_rec else ""}'
            print(log, flush=True)
        # print log

        # if iter == len(train_loader)-1:
        #     gs = epoch*1000 + iter
    gs = epoch
    writer.add_scalar('Train-LR', optimizer.param_groups[0]["lr"], global_step=gs)
    writer.add_scalar('Train-Loss', losses.avg, global_step=gs)
    writer.add_scalar('Train-Loss-text', losses_text.avg, global_step=gs)
    writer.add_scalar('Train-Loss-kernel', losses_kernels.avg, global_step=gs)
    writer.add_scalar('Train-Loss-emb', losses_emb.avg, global_step=gs)
    writer.add_scalar('Train-Loss-rec', losses_rec.avg, global_step=gs)
    writer.add_scalar('Train-IoU(text)', ious_text.avg, global_step=gs)
    writer.add_scalar('Train-IoU(kernel)', ious_kernel.avg, global_step=gs)
    writer.add_scalar('Train-ACC rec', accs_rec.avg, global_step=gs)
            

def adjust_learning_rate(optimizer, dataloader, epoch, iter, cfg):
    schedule = cfg.train_cfg.schedule

    if isinstance(schedule, str):
        assert schedule == 'polylr', 'Error: schedule should be polylr!'
        cur_iter = epoch * len(dataloader) + iter
        max_iter_num = cfg.train_cfg.epoch * len(dataloader)
        lr = cfg.train_cfg.lr * (1.0 - float(cur_iter) / max_iter_num) ** 0.9
    elif isinstance(schedule, tuple):
        lr = cfg.train_cfg.lr
        for i in range(len(schedule)):
            if epoch < schedule[i]:
                break
            lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, checkpoint_path, cfg):
    file_path = osp.join(checkpoint_path, 'checkpoint.pth.tar')
    torch.save(state, file_path)

    if cfg.data.train.type in ['synth'] or \
            (state['iter'] == 0 and
             state['epoch'] % 5 == 0):
        file_name = 'checkpoint_%dep.pth.tar' % state['epoch']
        file_path = osp.join(checkpoint_path, file_name)
        torch.save(state, file_path)


def main(args):
    cfg = Config.fromfile(args.config)
    cfg.update(dict(debug=args.debug))
    cfg.data.train.update(dict(debug=args.debug))
    cfg.report_speed = False
    #print log
    print(json.dumps(cfg._cfg_dict, indent=4))

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        cfg_name, _ = osp.splitext(osp.basename(args.config))
        checkpoint_path = osp.join('checkpoints', cfg_name)
    if not osp.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    # data loader
    data_loader = build_data_loader(cfg.data.train)
    train_size = int(0.8 * len(data_loader))
    test_size = len(data_loader) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data_loader, [train_size, test_size])
    print(f'data_loader_type:{type(data_loader)}')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=not cfg.debug,
        num_workers=4,
        drop_last=True,
        pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True,
        pin_memory=True)

    # model
    if hasattr(cfg.model, 'recognition_head'):
        cfg.model.recognition_head.update(
            dict(
                voc=data_loader.voc,
                char2id=data_loader.char2id,
                id2char=data_loader.id2char,
            ))
    model = build_model(cfg.model)

    if cfg.debug:
        # from IPython import embed; embed()
        checkpoint = torch.load('checkpoints/tmp.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])

    model = torch.nn.DataParallel(model).cuda()

    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        if cfg.train_cfg.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=cfg.train_cfg.lr,
                                        momentum=0.99,
                                        weight_decay=5e-4)
        elif cfg.train_cfg.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=0.001)#cfg.train_cfg.lr

    start_epoch = 0
    start_iter = 0
    if hasattr(cfg.train_cfg, 'pretrain'):
        assert osp.isfile(
            cfg.train_cfg.pretrain), 'Error: no pretrained weights found!'
        print('Finetuning from pretrained model %s.' % cfg.train_cfg.pretrain)
        checkpoint = torch.load(cfg.train_cfg.pretrain)
        model.load_state_dict(checkpoint['state_dict'])
    if args.resume:
        assert osp.isfile(args.resume), 'Error: no checkpoint directory found!'
        print('Resuming from checkpoint %s.' % args.resume)
        checkpoint = torch.load(args.resume)
        # print(checkpoint)
        start_epoch = checkpoint['epoch']
        start_iter = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    for epoch in range(start_epoch, cfg.train_cfg.epoch):
        print('\nEpoch: [%d | %d]' % (epoch + 1, cfg.train_cfg.epoch))
        # eval(eval_loader, model,  epoch, cfg)
        train(train_loader, model, optimizer, epoch, start_iter, cfg)
        if epoch%5==0:
            eval(eval_loader, model,  epoch, cfg)
        state = dict(epoch=epoch + 1,
                     iter=0,
                     state_dict=model.state_dict(),
                     optimizer=optimizer.state_dict())
        
        save_checkpoint(state, checkpoint_path, cfg=cfg)


if __name__ == '__main__':
    try:
        writer = init_tensorboard('./tblogs')
        parser = argparse.ArgumentParser(description='Hyperparams')
        parser.add_argument('--config', help='config file path',default='config\pan_pp\R18-AUG.py')
        parser.add_argument('--checkpoint', nargs='?', type=str, default=None)
        parser.add_argument('--resume', nargs='?', type=str, default=None)
        parser.add_argument('--debug', action='store_true')
        args = parser.parse_args()
        main(args)
    finally:
        writer.close()
