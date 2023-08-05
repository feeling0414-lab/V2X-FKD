import torch
from torch import nn

from .base import spconv, post_act_block, post_act_block_dense
from .norm import build_norm_layer


class SpMiddlePillarEncoderVgg(nn.Module):
    def __init__(
        self, model_cfg, in_planes=32, name="SpMiddlePillarEncoderVgg", **kwargs):
        super(SpMiddlePillarEncoderVgg, self).__init__()
        self.name = name
        self.nx, self.ny, self.nz = model_cfg['grid_size']

        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        block = post_t_block
        dense_block = post_act_block_dense

        self.conv1 = spconv.SparseSequential(
            spconv.SubMConv2d(64, 32, 3, padding=1, bias=False, indice_key="subm1"),
            build_norm_layer(norm_cfg, 32)[1],
            block(32, 32, 3, norm_cfg=norm_cfg, indice_key="subm1"),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(32, 64, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(64, 64, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),


        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(64, 128, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(128, 128, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),


        )

        self.conv4 = spconv.SparseSequential(

            block(128, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(256, 256, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),


        )
        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            dense_block(256, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            dense_block(256, 256, 3, norm_cfg=norm_cfg, padding=1),


        )

        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256,
        }
        self.backbone_strides = {
            'x_conv1': 1,
            'x_conv2': 2,
            'x_conv3': 4,
            'x_conv4': 8,
            'x_conv5': 16,
        }

    def forward(self, batch_dict):

        # 删除z轴
        voxel_coords = batch_dict['voxel_coords']
        xy = voxel_coords[:,2:].contiguous()
        index = voxel_coords[:,0].unsqueeze(dim=1).contiguous()
        pillar_coords =torch.cat((index, xy),dim=1)
        #获得batch—size
        spatial_batch_size = voxel_coords[:, 0].max().int().item() + 1

        # spatial_features = batch_dict['spatial_features']
        input_sp_tensor = spconv.SparseConvTensor(
            features = batch_dict.get("pillar_features"),
            indices = pillar_coords.int(),
            spatial_shape = [self.ny,self.nx],
            batch_size = spatial_batch_size
        )
        x_conv1 = self.conv1(input_sp_tensor)    #[192,704] <-[192,704]
        x_conv2 = self.conv2(x_conv1)            #[96,352] <-[192,704]
        x_conv3 = self.conv3(x_conv2)            #[48,176] <-[96,352]
        x_conv4 = self.conv4(x_conv3)            #[24,88] <-[48,176]
        x_conv4 = x_conv4.dense()                #[24,88] <-[24,88]
        x_conv5 = self.conv5(x_conv4)            #[12,44] <-[24,88]
        x_conv3 = x_conv3.dense()
        batch_dict['encode_feature'] = (x_conv3,x_conv4, x_conv5)
        return batch_dict

class PillarEncoderDistillation(nn.Module):
    def __init__(
        self, model_cfg, in_planes=32, name="PillarEncoderDistillation", **kwargs):
        super(PillarEncoderDistillation, self).__init__()
        self.name = name
        self.nx, self.ny, self.nz = model_cfg['grid_size']

        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        block = post_act_block
        dense_block = post_act_block_dense

        self.conv1 = spconv.SparseSequential(
            spconv.SubMConv2d(64, 32, 3, padding=1, bias=False, indice_key="subm1"),
            build_norm_layer(norm_cfg, 32)[1],
            block(32, 32, 3, norm_cfg=norm_cfg, indice_key="subm1"),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(32, 64, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(64, 64, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
            # block(64, 64, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),

        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(64, 128, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(128, 128, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            block(128, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(256, 256, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
        )
        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            dense_block(256, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            dense_block(256, 256, 3, norm_cfg=norm_cfg, padding=1),


        )

        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256,
        }
        self.backbone_strides = {
            'x_conv1': 1,
            'x_conv2': 2,
            'x_conv3': 4,
            'x_conv4': 8,
            'x_conv5': 16,
        }

    def forward(self, batch_dict):

        # 删除z轴
        voxel_coords = batch_dict['voxel_coords']
        xy = voxel_coords[:,2:].contiguous()
        index = voxel_coords[:,0].unsqueeze(dim=1).contiguous()
        pillar_coords =torch.cat((index, xy),dim=1)
        #获得batch—size
        spatial_batch_size = voxel_coords[:, 0].max().int().item() + 1

        # spatial_features = batch_dict['spatial_features']
        input_sp_tensor = spconv.SparseConvTensor(
            features = batch_dict.get("pillar_features"),
            indices = pillar_coords.int(),
            spatial_shape = [self.ny,self.nx],
            batch_size = spatial_batch_size
        )
        x_conv1 = self.conv1(input_sp_tensor)    #[192,704] <-[192,704]
        x_conv2 = self.conv2(x_conv1)            #[96,352] <-[192,704]
        x_conv3 = self.conv3(x_conv2)            #[48,176] <-[96,352]
        x_conv4 = self.conv4(x_conv3)            #[24,88] <-[48,176]
        x_conv4 = x_conv4.dense()                #[24,88] <-[24,88]
        x_conv5 = self.conv5(x_conv4)            #[12,44] <-[24,88]
        x_conv3 = x_conv3.dense()
        batch_dict['encode_feature'] = (x_conv3,x_conv4, x_conv5)
        return batch_dict