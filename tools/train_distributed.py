import argparse
import os
import statistics

import torch
import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import v2xvit.hypes_yaml.yaml_utils as yaml_utils
from v2xvit.tools import train_utils
from v2xvit.data_utils.datasets import build_dataset
import torch.nn as nn
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

#计算参数量
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def train_parser():

    parser = argparse.ArgumentParser(description="synthetic data generation")#合成数据生成
    parser.add_argument("--local_rank", default=-1, help="local device id on current node", type=int)

    parser.add_argument("--hypes_yaml", type=str,
                        default="/media/wanghai/NewSSD/HCZ/v2x-vit_myself/v2xvit/hypes_yaml/distillation_v2xvit.yaml",
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true', help="whether train with half precision")#是否使用半精度
    opt = parser.parse_args()
    return opt

    #加入local_rank


def main(local_rank,hypes):

    model = train_utils.create_model(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)
    model = torch.nn.parallel.DistributedDataParallel(model.cuda(local_rank),device_ids=[local_rank],
                                                      find_unused_parameters=True)#拷贝模型

    train_sampler = torch.utils.data.DistributedSampler(opencood_train_dataset)
    train_data_loader = torch.utils.data.DataLoader(
        opencood_train_dataset, batch_size = hypes['train_params']['batch_size'],num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                sampler=train_sampler)
    val_data_loader = torch.utils.data.DataLoader(
        opencood_validate_dataset, batch_size=hypes['train_params']['batch_size'],num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train)


    # define the loss
    criterion = train_utils.create_loss(hypes)
    # lr scheduler setup
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer)
    init_epoch = 0
    if local_rank==0:
        saved_path = train_utils.setup_train(hypes)
        # record training
        writer = SummaryWriter(saved_path)
    # half precision training自动混合精度训练，节省显存并加快推理速度
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate
    for epoch in range(init_epoch, max(epoches, init_epoch)):

        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        #进度条
        if local_rank == 0:
            pbar2 = tqdm.tqdm(total=len(train_data_loader), leave=True)

        train_sampler.set_epoch(epoch)#使得每张卡在每个周期的数据都是随机的

        for i, batch_data in enumerate(train_data_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, local_rank)
            # case1 : late fusion train --> only ego needed
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                # first argument is always your output dictionary,第一参数为模型输出
                # second argument is always your label dictionary.第二参数为标签
                final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            if local_rank == 0:
                criterion.logging(epoch, i, len(train_data_loader), writer, pbar=pbar2)
                pbar2.update(1)
            # back-propagation
            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()
        #验证模型
        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_data_loader):
                    model.eval()
                    batch_data = train_utils.to_device(batch_data, local_rank)
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                               valid_ave_loss))
            if local_rank == 0:
                writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)


        #保存模型
        if epoch % hypes['train_params']['save_freq'] == 0 and local_rank == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        scheduler.step(epoch)
    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':

    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    n_gpus = 3 # GPU的数量
    torch.distributed.init_process_group("nccl", world_size=n_gpus, rank = opt.local_rank)

    torch.cuda.set_device(opt.local_rank)#修改环境变量


    print('%s GPU Dataset Building' % opt.local_rank)


    main(opt.local_rank,hypes)
