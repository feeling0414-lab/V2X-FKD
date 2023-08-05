import numpy as np
import torch
import torch.nn as nn


class DistillationBase(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #                            nn.BatchNorm2d(128),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #                            nn.BatchNorm2d(128),
        #                            nn.ReLU(inplace=True), )
        # self.conv2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        #                            nn.BatchNorm2d(256),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #                            nn.BatchNorm2d(256),
        #                            nn.ReLU(inplace=True), )
        # self.conv3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        #                            nn.BatchNorm2d(512),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #                            nn.BatchNorm2d(512),
        #                            nn.ReLU(inplace=True), )
        # self.conv4 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #                            nn.BatchNorm2d(512),
        #                            nn.ReLU(inplace=True),
        #                            )
        #
        # self.conv6 = nn.Sequential(nn.Conv2d(768, 384, kernel_size=3, stride=2, padding=1),
        #                            nn.BatchNorm2d(384),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
        #                            nn.BatchNorm2d(384),
        #                            nn.ReLU(inplace=True), )
        #
        self.conv1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),)
        self.conv3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),)
        self.conv6 = nn.Sequential(nn.Conv2d(768, 256, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(384),
                                    nn.ReLU(inplace=True),)



    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        x = spatial_features
        # conv1 = self.conv1(x)#64->128
        # conv2 = self.conv2(conv1)#128->256
        # conv3 = self.conv3(conv2)#256->512
        # conv4 = self.conv4(conv3)#512->1024->512
        # # conv5 = self.conv5(torch.cat((conv4,conv3),dim=1)) #1024+512=1536->512
        # conv6 = self.conv6(torch.cat((conv4,conv2),dim=1)) #512+256=768->256

        conv1 = self.conv1(x)#64->128
        conv2 = self.conv2(conv1)#128->256
        conv3 = self.conv3(conv2)#256->512
        conv6 = self.conv6(torch.cat((conv3,conv2),dim=1)) #512+256=768->384

        data_dict['spatial_features_2d'] = conv6
        return data_dict
