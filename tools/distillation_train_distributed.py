import argparse
import os


import torch
import tqdm
import statistics

import v2xvit.hypes_yaml.yaml_utils as yaml_utils
from v2xvit.tools import train_utils
from v2xvit.data_utils.datasets import build_dataset,build_teacher_dataset,build_distillation_dataset
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
#计算参数量
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")#合成数据生成
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true', help="whether train with half precision")#是否使用半精度
    opt = parser.parse_args()
    return opt



def main(local_rank,hypes):

    print('Creating Model')
    model_student = train_utils.create_model(hypes)
    model_teacher = train_utils.create_teacher_model(hypes)
    # optimizer setup
    optimizer_student = train_utils.setup_optimizer(hypes, model_student)
    optimizer_teacher = train_utils.setup_optimizer(hypes, model_teacher)
    print('Dataset Building')
    combine_train_dataset =build_distillation_dataset(hypes,visualize=False, train=True)
    val_dataset = build_dataset(hypes,visualize=False,train=False)

    model_student = torch.nn.parallel.DistributedDataParallel(model.cuda(local_rank),device_ids=[local_rank],
                                                      find_unused_parameters=True)#拷贝模型
    model_teacher = torch.nn.parallel.DistributedDataParallel(model.cuda(local_rank),device_ids=[local_rank],
                                                      find_unused_parameters=True)#拷贝模型
    train_sampler = torch.utils.data.DistributedSampler(combine_train_dataset)
    train_data_loader = torch.utils.data.DataLoader(
        combine_train_dataset,batch_size = hypes['train_params']['batch_size'],num_workers=8,
                                collate_fn=combine_train_dataset.collate_batch_train,
                                sampler=train_sampler)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=hypes['train_params']['batch_size'],num_workers=8,
                                collate_fn=val_dataset.collate_batch_train)
    # define the loss
    criterion_combine = train_utils.create_combine_loss(hypes)
    criterion=train_utils.create_loss(hypes)
    # lr scheduler setup
    scheduler_student = train_utils.setup_lr_schedular(hypes, optimizer_student)
    scheduler_teacher = train_utils.setup_lr_schedular(hypes, optimizer_teacher)

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

        for param_group in optimizer_student.param_groups:
            print('learning rate %f' % param_group["lr"])
        #进度条
        if local_rank == 0:
            pbar2 = tqdm.tqdm(total=len(train_data_loader), leave=True)

        train_sampler.set_epoch(epoch)#使得每张卡在每个周期的数据都是随机的

        for i, batch_data in enumerate(train_data_loader):
            # the model will be evaluation mode during validation
            model_student.train()
            model_student.zero_grad()
            optimizer_student.zero_grad()

            model_teacher.train()
            model_teacher.zero_grad()
            optimizer_teacher.zero_grad()

            batch_data = train_utils.to_device(batch_data, local_rank)
            # case1 : late fusion train --> only ego needed
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not opt.half:
                student_dict, teacher_dict, output_dict = \
                    train_utils.combineTrain(model_student, model_teacher, batch_data)
                final_loss = criterion_combine(teacher_dict, student_dict, output_dict,
                                               batch_data[0]['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    student_dict, teacher_dict, output_dict = \
                        train_utils.combineTrain(model_student, model_teacher, batch_data)
                    final_loss = criterion_combine(teacher_dict, student_dict, output_dict,
                                                   batch_data[0]['ego']['label_dict'])
            if local_rank == 0:
                criterion.logging(epoch, i, len(combine_train_loader), writer, pbar=pbar2)
                pbar2.update(1)
            # back-propagation

            if not opt.half:
                final_loss.backward()
                optimizer_student.step()
                optimizer_teacher.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer_student)
                scaler.step(optimizer_teacher)
                scaler.update()
        #验证模型
        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_data_loader):
                    model_student.eval()
                    batch_data = train_utils.to_device(batch_data, local_rank)
                    student_dict,output_dict = model_student(batch_data['ego'])
                    final_loss = criterion(output_dict, batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            if local_rank == 0:
                writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)


        #保存模型
        if epoch % hypes['train_params']['save_freq'] == 0 and local_rank == 0:
            torch.save([model_student.state_dict(),
                       model_teacher.state_dict()],
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