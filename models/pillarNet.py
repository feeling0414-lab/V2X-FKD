"""
Vanilla pillarNet for early and late fusion.
"""
import torch.nn as nn
import torch
import spconv.pytorch as spconv
from v2xvit.models.sub_modules.pillar_vfe import PillarVFE
from v2xvit.models.sub_modules.point_pillar_scatter import PointPillarScatter
from v2xvit.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from v2xvit.models.sub_modules.downsample_conv import DownsampleConv
from v2xvit.models.sub_modules.pcnvgg import SpMiddlePillarEncoderVgg
from v2xvit.models.sub_modules.neck import Neck


class PillarNet(nn.Module):
    def __init__(self, args):
        super(PillarNet, self).__init__()

        #PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.shape_size = args['point_pillar_scatter']['grid_size']
        self.Encoder = SpMiddlePillarEncoderVgg(args['point_pillar_scatter'])
        self.Neck = Neck()
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        self.cls_head = nn.Conv2d(args['cls_head_dim'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['cls_head_dim'],
                                  7 * args['anchor_number'],
                                  kernel_size=1)

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'batch_size': data_dict['object_bbx_center'].size(0)}

        batch_dict = self.pillar_vfe(batch_dict) #pillar_features
        #pillarNet
        batch_dict = self.Encoder(batch_dict)
        spatial_features_2d = self.Neck(batch_dict)

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict


