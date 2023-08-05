import os
from collections import OrderedDict

import numpy as np
import torch
import matplotlib.pyplot as plt
from v2xvit.utils.common_utils import torch_tensor_to_numpy


def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


# distailltion edition
def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']
    output = model(cav_content)
    #distillation edition
    if type(output) is dict:
        output_dict['ego'] = output
    else:
        # _, output_dict['ego'] = output
        attention_map, output_dict['ego'] = output

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return  attention_map,pred_box_tensor, pred_score, gt_box_tensor,output_dict['ego']
    # return  pred_box_tensor, pred_score, gt_box_tensor,output_dict['ego']

# def inference_early_fusion(batch_data, model, dataset):
#     """
#     Model inference for early fusion.
#
#     Parameters
#     ----------
#     batch_data : dict
#     model : opencood.object
#     dataset : opencood.EarlyFusionDataset
#
#     Returns
#     -------
#     pred_box_tensor : torch.Tensor
#         The tensor of prediction bounding box after NMS.
#     gt_box_tensor : torch.Tensor
#         The tensor of gt bounding box.
#     """
#     output_dict = OrderedDict()
#     cav_content = batch_data['ego']
#
#     output_dict['ego'] = model(cav_content)
#     pred_box_tensor, pred_score, gt_box_tensor = \
#         dataset.post_process(batch_data,
#                              output_dict)
#
#     return pred_box_tensor, pred_score, gt_box_tensor,output_dict['ego']


def draw_attention_mid(map,address,address_num):
    map = map['rm'][0]
    for i in range(3):
        plt.figure(figsize=(16, 4), dpi=400)
        # fig, ax = plt.subplots(1, 1, figsize=(15, 5), dpi=400)
        x = map[i].cpu()
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        plt.imshow(x)
        ax = plt.gca()
        ax.invert_xaxis()
        vis_save_path = os.path.join(address, '%5d_%d_heatmap.png' % (address_num, i))
        # plt.colorbar()
        # cb.set_label('colormaping')
        plt.savefig(vis_save_path)

        plt.show()
    pass


def draw_heat_map(map,address,address_num):

    # fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(15, 12), dpi=500)
    # x1 = map[0][0].cpu()
    # x2 = map[1][0].cpu()
    # x3 = map[2][0].cpu()
    # # plt.imshow(x, cmap=plt.cm.hot)
    # ax1.contourf(x1,cmap="viridis")
    # ax1.invert_xaxis()
    # ax2.contourf(x2,cmap="viridis")
    # ax2.invert_xaxis()
    # ax3.contourf(x3,cmap="viridis")
    # ax3.invert_xaxis()


    car_num=map.shape[0]
    if car_num == 1:
        fig, ax = plt.subplots(car_num, 1, figsize=(5 * car_num, 4 * car_num), dpi=400)
        x = map[0][0].cpu()
        ax.contourf(x, cmap="viridis")
        ax.invert_xaxis()
    else:
        for i in range(car_num):
            plt.figure(figsize=(16, 4), dpi=400)
            # fig, ax = plt.subplots(1, 1, figsize=(15, 5), dpi=400)
            x = map[i][0].cpu()
            x = (x-torch.min(x))/(torch.max(x)-torch.min(x))
            plt.contourf(x,cmap="viridis")

            ax = plt.gca()
            ax.invert_xaxis()
            vis_save_path = os.path.join(address,'%5d_%d_heatmap.png' % (address_num, i))
            plt.colorbar()
            # cb.set_label('colormaping')
            plt.savefig(vis_save_path)

            plt.show()
    car_num = map.shape[0]
    pass


def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return inference_early_fusion(batch_data, model, dataset)


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy' % timestamp), gt_np)
