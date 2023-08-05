import torch
import torch.nn as nn

from v2xvit.models.sub_modules.pillar_vfe import PillarVFE
from v2xvit.models.sub_modules.point_pillar_scatter import PointPillarScatter
from v2xvit.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from v2xvit.models.sub_modules.fuse_utils import regroup
from v2xvit.models.sub_modules.downsample_conv import DownsampleConv
from v2xvit.models.sub_modules.naive_compress import NaiveCompressor
from v2xvit.models.sub_modules.v2xvit_basic import V2XTransformer
from v2xvit.models.sub_modules.distillation_base import DistillationBase
from v2xvit.models.sub_modules.pcnvgg import SpMiddlePillarEncoderVgg,PillarEncoderDistillation
from v2xvit.models.sub_modules.neck_ViT import Neck_ViT
from v2xvit.models.sub_modules.v2v_fuse import V2VNetFusion

from v2xvit.models.sub_modules.self_attn import AttFusion



class PillarNetDistillationV2xvit(nn.Module):
    def __init__(self, args):
        super(PillarNetDistillationV2xvit, self).__init__()

        self.max_cav = args['max_cav']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.shape_size = args['pillarNet_student']['grid_size']
        self.Encoder = PillarEncoderDistillation(args['pillarNet_student'])
        self.Neck = Neck_ViT()

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False#缩小头
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.fusion_net = V2XTransformer(args['transformer'])

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def unpad_prior_encoding(self, x, record_len):
        # remove padded zeros to form tensor with shape (N, 3)
        # x: (B, L, 3); record_len: (B)
        B = x.shape[0]
        out = []
        for i in range(B):
            # (valid_len, 3)
            out.append(x[i, :record_len[i], :])
        out = torch.cat(out, dim=0)
        # (N, 3)
        return out

    def pickup(self,batch_dict,record_len):
        spatial_feature= batch_dict['encode_feature'][2]
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(spatial_feature, cum_sum_len[:-1].cpu())
        pickup_tensor = []
        for xx in split_x:
            pickup_tensor.append(xx[0].unsqueeze(0))
        ego_tensor = torch.cat(pickup_tensor,dim=0)
        return ego_tensor

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']#体素特征 [X,32,4]
        voxel_coords = data_dict['processed_lidar']['voxel_coords'] #体素坐标 [X,4]
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points'] #体素坐标 [X]
        record_len = data_dict['record_len'] #记录长度
        spatial_correction_matrix = data_dict['spatial_correction_matrix']#空间矫正矩阵
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        # B, max_cav, 3(dt dv infra), 1, 1
        prior_encoding =\
            data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)#增加两个维度

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'batch_size': data_dict['object_bbx_center'].size(0),
                      'record_len': record_len}
        # n, 4 -> n, c 柱状特征提取
        batch_dict = self.pillar_vfe(batch_dict)
        # # # n, c -> N, C, H, W 放回到原始场景中
        # batch_dict = self.scatter(batch_dict)
        batch_dict = self.Encoder(batch_dict)
        spatial_features_2d = self.Neck(batch_dict)

        # downsample feature to reduce memory# 降低采样特征用于减少存储
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        #ViT
        # N, C, H, W -> B,  L, C, H, W
        regroup_feature, mask = regroup(spatial_features_2d,
                                        record_len,
                                        self.max_cav)
        # prior encoding added
        prior_encoding = prior_encoding.repeat(1, 1, 1,
                                               regroup_feature.shape[3],
                                               regroup_feature.shape[4])
        #在第3个维度拼接
        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)

        # b l c h w -> b l h w c
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        # transformer fusion
        fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
        # b h w c -> b c h w
        fused_feature = fused_feature.permute(0, 3, 1, 2)
        middle_feature = self.pickup(batch_dict,record_len)


        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        return middle_feature,output_dict
