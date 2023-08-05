"""
Vanilla pillarNet for early and late fusion.
"""
import torch.nn as nn
import torch
import spconv.pytorch as spconv
from v2xvit.models.sub_modules.pillar_vfe import PillarVFE
from v2xvit.models.sub_modules.downsample_conv import DownsampleConv
from v2xvit.models.sub_modules.pcnvgg import SpMiddlePillarEncoderVgg



class PillarNetTeacher(nn.Module):
    def __init__(self, args):
        super(PillarNetTeacher, self).__init__()

        #PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.shape_size = args['pillarNet_teacher']['grid_size']
        self.Encoder = SpMiddlePillarEncoderVgg(args['pillarNet_teacher'])
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
        batch_dict = self.Encoder(batch_dict)


        return batch_dict['encode_feature'][2]

