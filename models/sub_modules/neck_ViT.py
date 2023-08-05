import torch
from torch import nn

class Neck_ViT(nn.Module):
    def __init__(self):
        super(Neck_ViT, self).__init__()
        self.UPConv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()

        )
        self.UPConv2  = nn.Sequential(
            nn.ConvTranspose2d(384,128,kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()

        )
        self.UPConv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )


    def forward(self,batch_dict):
        Conv_3, Conv_4, Conv_5 = batch_dict['encode_feature']
        Conv_5 = self.UPConv1(Conv_5)
        connect2 = self.UPConv2(torch.cat((Conv_4,Conv_5),dim=1))
        final = self.UPConv3(torch.cat((Conv_3,connect2), dim=1))
        return  final


# class Neck(nn.Module):
#     def __init__(self):
#         super(Neck, self).__init__()
#         self.UPConv1 = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )
#
#         self.UPConv2 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
#
#
#
#
#     def forward(self,Conv_4,Conv_5):
#         Conv_5 = self.UPConv1(Conv_5)
#         #4和5连接
#         connect1 = torch.cat((Conv_4,Conv_5),dim=1)
#         final = self.UPConv2(connect1)
#         return  final