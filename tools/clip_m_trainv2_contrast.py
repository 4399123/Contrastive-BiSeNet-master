#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '..')
import os
import os.path as osp
import random
import logging
import time
import json
import argparse
import numpy as np
from tabulate import tabulate
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.cuda.amp as amp

from lib.models import model_factory
from configs import set_cfg_from_file
from lib.data import get_data_loader
from evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss,FocalLoss,PolyFocalLoss,IoULoss,DiceLoss
from lib.ohem_ce_loss import DiceWithOhemCELoss,DiceWithFocalLoss,DiceBCELoss,OhemWithIoULoss,OhemWithFocalLoss,GDiceWithOhemCELoss,LogCoshDiceLossWithOhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter,AverageMeter
from lib.logger import setup_logger, print_log_msg,print_log_msgs
from tqdm import tqdm
from timm.optim import AdamW,AdamP,RAdam,Lookahead,AdaBelief,Lion,Lamb
from timm.scheduler import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter
from timm.layers.norm_act import convert_sync_batchnorm
from lib.losses import ContrastCELoss
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4"

## fix all random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
#  torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')
printlabels=['background','QPZZ','MDBD','MNYW','WW','LMPS','BMQQ','LMHH','KTAK']

def parse_args():
    parse = argparse.ArgumentParser()
    # parse.add_argument('--config', dest='config', type=str,
    #         default='../configs/bisenetv1_blueface_caformer_s36.py',)
    parse.add_argument('--config', dest='config', type=str,
            default='../configs/bisenetv1_blueface_efficientnet_b3.py',)
    parse.add_argument('--finetune-from', type=str, default=None,)
    parse.add_argument("--local_rank", type=int)
    return parse.parse_args()

args = parse_args()
print('Loading configuration:{}'.format(args.config))
cfg = set_cfg_from_file(args.config)

def set_model(lb_ignore=255):
    logger = logging.getLogger()
    net = model_factory[cfg.model_type](n_classes=cfg.n_cats,use_fp16=cfg.use_fp16,aux_mode='train')
    if not args.finetune_from is None:
        logger.info(f'load pretrained weights from {args.finetune_from}')
        msg = net.load_state_dict(torch.load(args.finetune_from,
            map_location='cpu'), strict=False)
        logger.info('\tmissing keys: ' + json.dumps(msg.missing_keys))
        logger.info('\tunexpected keys: ' + json.dumps(msg.unexpected_keys))
    # if cfg.use_sync_bn: net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    if cfg.use_sync_bn: net = convert_sync_batchnorm(net)
    net.cuda()
    net.train()

    loss_opt=0

    criteria_pre=0
    criteria_aux=0
    if(loss_opt==0):
        criteria_pre = ContrastCELoss(class_weights=None)
        criteria_aux = [OhemCELoss(0.7, lb_ignore) for _ in range(cfg.num_aux_heads)]
    elif(loss_opt==1):
        criteria_pre = DiceWithOhemCELoss()
        criteria_aux = [DiceWithOhemCELoss() for _ in range(cfg.num_aux_heads)]
    elif(loss_opt==2):
        criteria_pre = DiceWithFocalLoss()
        criteria_aux = [DiceWithFocalLoss() for _ in range(cfg.num_aux_heads)]
    elif (loss_opt ==3):
        criteria_pre = OhemWithIoULoss()
        criteria_aux = [OhemWithIoULoss() for _ in range(cfg.num_aux_heads)]
    elif (loss_opt == 4):
        criteria_pre = OhemWithFocalLoss()
        criteria_aux = [OhemWithFocalLoss() for _ in range(cfg.num_aux_heads)]
    elif (loss_opt ==5):
        criteria_pre = DiceBCELoss()
        criteria_aux = [DiceBCELoss() for _ in range(cfg.num_aux_heads)]
    elif (loss_opt ==6):
        criteria_pre = GDiceWithOhemCELoss()
        criteria_aux = [GDiceWithOhemCELoss() for _ in range(cfg.num_aux_heads)]
    elif (loss_opt ==7):
        criteria_pre = LogCoshDiceLossWithOhemCELoss()
        criteria_aux = [LogCoshDiceLossWithOhemCELoss() for _ in range(cfg.num_aux_heads)]
    else:
        print('no such loss !!!')


    return net, criteria_pre, criteria_aux


def set_optimizer(model):

    optim_opt=2
    optim=0
    if(optim_opt==0):
        optim = AdaBelief(model.parameters(), lr=cfg.lr_start, weight_decay=cfg.weight_decay)
    elif (optim_opt==1):
        optim =AdamP(model.parameters(),lr=cfg.lr_start,weight_decay=cfg.weight_decay,nesterov=True)
    elif (optim_opt ==2):
        optim = AdamW(model.parameters(), lr=cfg.lr_start, weight_decay=cfg.weight_decay)
    elif (optim_opt ==3):
        optim = torch.optim.SGD(model.parameters(),lr=cfg.lr_start,momentum=0.9,weight_decay=cfg.weight_decay,)
    elif (optim_opt ==4):
        optim = Lion(model.parameters(),lr=cfg.lr_start,weight_decay=cfg.weight_decay,)
    elif (optim_opt ==5):
        optim = Lamb(model.parameters(),lr=cfg.lr_start,weight_decay=cfg.weight_decay,)
    # base_optim = RAdam(model.parameters(), lr=cfg.lr_start, weight_decay=5e-4)
    # optim=Lookahead(base_optimizer=base_optim,k=5,alpha=0.5)

    return optim


def set_model_dist(net):
    # local_rank = int(os.environ['LOCAL_RANK'])
    local_rank = int(args.local_rank)
    # local_rank = 0
    net = nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank, ],
        find_unused_parameters=True,
        output_device=local_rank
        )
    return net


def set_meters():
    # time_meter = TimeMeter(cfg.max_iter)
    time_meter = TimeMeter(cfg.max_epochs)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters

def train(writer):
    logger = logging.getLogger()

    ## dataset
    dl = get_data_loader(cfg, mode='train')

    ## model
    net, criteria_pre, criteria_aux = set_model(dl.dataset.lb_ignore)

    ## optimizer
    optim = set_optimizer(net)

    ## mixed precision training
    scaler = amp.GradScaler()

    ## ddp training
    net = set_model_dist(net)

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()

    lr_schdr = CosineLRScheduler(optimizer=optim,
                                 t_initial=cfg.max_epochs,
                                 lr_min=5e-6,
                                 # warmup_t=0.05 * cfg.max_epochs,
                                 warmup_t=5,
                                 warmup_lr_init=1e-5)
    # lr_schdr = CosineLRScheduler(optimizer=optim,
    #                              t_initial=cfg.max_epochs//2,
    #                              cycle_limit=2,
    #                              cycle_decay=0.5,
    #                              lr_min=9.5e-5,
    #                              warmup_t=5,
    #                              warmup_lr_init=1e-4)
    miou=0.0
    mprecision=0.0
    mrecall=0.0
    gap=int(len(dl)/10)
    if(gap==0): gap=2
    ## train loop
    for epoch in range(cfg.max_epochs):

        lr_schdr.step(epoch)
        lr = optim.param_groups[0]['lr']
        writer.add_scalar('lr', lr, epoch)

        for it, (im, lb) in enumerate(dl):
            im = im.cuda()
            lb = lb.cuda()

            lb = torch.squeeze(lb, 1)

            optim.zero_grad()
            with amp.autocast(enabled=cfg.use_fp16):
                logits, *logits_aux = net(im)
                loss_pre,partial_losses = criteria_pre(logits, lb)
                loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
                loss = loss_pre +0.4*sum(loss_aux)

            scaler.scale(loss).backward()

            scaler.step(optim)
            scaler.update()
            torch.cuda.synchronize()

            # time_meter.update()
            loss_meter.update(loss.item())
            loss_pre_meter.update(loss_pre.item())
            _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]

            ## print training log message
            # if (it + 1) % 200 == 0:
            if (it + 1) % gap == 0:
                # print_log_msg(epoch,cfg.max_epochs,it, len(dl), lr, time_meter, loss_meter,loss_pre_meter, loss_aux_meters)
                print_log_msgs(epoch, cfg.max_epochs, it, len(dl), lr, loss_meter, loss_pre_meter,loss_aux_meters,writer)
        interv,ets=time_meter.get()
        logger.info('ets:{},interv:{:.2f}s'.format(ets,interv))
        time_meter.update()

        torch.cuda.empty_cache()
        iou_heads, iou_content, f1_heads, f1_content,precision_heads, precision_content,recall_heads, recall_content = eval_model(cfg, net.module,printlabels)

        logger.info('\neval results of miou metric:')
        logger.info('\n' + tabulate(iou_content, headers=iou_heads, tablefmt='orgtbl'))

        logger.info('\neval results of mprecision metric:')
        logger.info('\n' + tabulate(precision_content, headers=precision_heads, tablefmt='orgtbl'))

        logger.info('\neval results of mrecall metric:')
        logger.info('\n' + tabulate(recall_content, headers=recall_heads, tablefmt='orgtbl'))

        writer.add_scalar('miou', float(iou_content[-2][-1]), epoch)
        writer.add_scalar('mprecision', float(precision_content[-1][-1]), epoch)
        writer.add_scalar('mrecall', float(recall_content[-1][-1]), epoch)
        if(miou<float(iou_content[-2][-1])):
            miou=float(iou_content[-2][-1])
            mprecision=float(precision_content[-1][-1])
            mrecall=float(recall_content[-1][-1])
            if dist.get_rank() == 0:torch.save(net.module.state_dict(),'../pt/best.pt')
            logger.info("miou:{},mprecision:{},mrecall:{},save model!!!".format(miou,mprecision,mrecall))
        logger.info("best miou:{},mprecision:{},mrecall:{}".format(miou,mprecision,mrecall))

    return


def main(writer):
    # local_rank = int(os.environ['LOCAL_RANK'])
    # local_rank = 0
    local_rank = int(args.local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger(f'{cfg.model_type}-{cfg.dataset.lower()}-train', cfg.respth)
    train(writer)


if __name__ == "__main__":
    writer = SummaryWriter()
    main(writer)
