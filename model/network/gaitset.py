import torch
import torch.nn as nn
import numpy as np

from .basic_blocks import SetBlock, BasicConv2d
from attention_augmented_conv import AugmentedConv
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

class SetNet(nn.Module):
    def __init__(self, hidden_dim):
        super(SetNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None

        _set_in_channels = 1
        _set_channels = [32, 64, 128]
        self.set_layer1 = SetBlock(BasicConv2d(_set_in_channels, _set_channels[0], 5, padding=2))
        self.set_layer2 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[0], 3, padding=1), True)
        self.set_layer3 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[1], 3, padding=1))
        self.set_layer4 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[1], 3, padding=1), True)
        self.set_layer5 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[2], 3, padding=1))
        self.set_layer6 = SetBlock(BasicConv2d(_set_channels[2], _set_channels[2], 3, padding=1))

        _gl_in_channels = 32
        _gl_channels = [64, 128]
        self.gl_layer1 = BasicConv2d(_gl_in_channels, _gl_channels[0], 3, padding=1)
        self.gl_layer2 = BasicConv2d(_gl_channels[0], _gl_channels[0], 3, padding=1)
        self.gl_layer3 = BasicConv2d(_gl_channels[0], _gl_channels[1], 3, padding=1)
        self.gl_layer4 = BasicConv2d(_gl_channels[1], _gl_channels[1], 3, padding=1)
        self.gl_pooling = nn.MaxPool2d(2)

        self.bin_num = [1, 2, 4, 8, 16]
        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_num) * 2, 128, hidden_dim)))])

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal(m.weight.data, 1.0, 0.02)
                nn.init.constant(m.bias.data, 0.0)

    def frame_max(self, x):
        if self.batch_frame is None:
            return torch.max(x, 1)
        else:
            _tmp = [
                torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
                ]
            max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return max_list, arg_max_list

    def frame_median(self, x):
        if self.batch_frame is None:
            return torch.median(x, 1)
        else:
            _tmp = [
                torch.median(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
                ]
            median_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_median_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return median_list, arg_median_list

    def forward(self, silho, batch_frame=None):
        # n: batch_size, s: frame_num, k: keypoints_num, c: channel
        if batch_frame is not None:
            batch_frame = batch_frame[0].data.cpu().numpy().tolist()
            _ = len(batch_frame)
            for i in range(len(batch_frame)):
                if batch_frame[-(i + 1)] != 0:
                    break
                else:
                    _ -= 1
            batch_frame = batch_frame[:_]
            frame_sum = np.sum(batch_frame)
            if frame_sum < silho.size(1):
                silho = silho[:, :frame_sum, :, :]
            self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
        n = silho.size(0) # silho torch.Size([32, 30, 64, 44])

        x = silho.unsqueeze(2) #torch.Size([32, 30, 1, 64, 44])
        del silho

        # 主干网卷积层1
        x = self.set_layer1(x)# 1：[bs, 30, 32, 64, 44] Conv
        x = self.set_layer2(x)# 2: [bs, 30, 32, 32, 22] Conv + Pooling
        # MGP卷积层1
        gl = self.gl_layer1(self.frame_max(x)[0])# 3: [bs, 64, 32, 22] SP + Conv
        gl = self.gl_layer2(gl)# 4: [bs, 64, 32, 22] Conv
        gl = self.gl_pooling(gl)# 5: [bs, 64, 16, 22] Pooling

        # 主干网卷积层2
        x = self.set_layer3(x)# 6: [bs, 30, 64, 32, 22] Conv
        x = self.set_layer4(x)# 7: [bs, 30, 64, 16, 11] Conv + Pooling
        # MGP卷积层2
        gl = self.gl_layer3(gl + self.frame_max(x)[0])# 8: [bs, 128, 16, 11] SP + Concat + Conv
        #gl = self.gl_layer4(gl)# 9: [bs, 128, 16, 11] Conv

        #Attention

        gl_atten = AugmentedConv(in_channels=128, out_channels=128, kernel_size=3, dk=40, dv=4, Nh=4, relative=False, stride=1).to(device)
        x_atten  = SetBlock(AugmentedConv(in_channels=64, out_channels=128, kernel_size=3, dk=40, dv=4, Nh=4, relative=False, stride=1)).to(device)

        gl = gl_atten(gl) # [bs, 128, 16, 11]


        # 主干网卷积层3
        #x = self.set_layer5(x)# 10: [bs, 30, 128, 16, 11] Conv
        #x = self.set_layer6(x)# 11: [bs, 30, 128, 16, 11] Conv
        x = x_atten(x) # [bs,30,128,16,11]

        x = self.frame_max(x)[0]# 12：[bs, 128, 16, 11] SP
        gl = gl + x# 13: [bs, 128, 16, 11] Concat

        feature = list()
        n, c, h, w = gl.size()
        for num_bin in self.bin_num:
            z = x.view(n, c, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)
            z = gl.view(n, c, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()

        feature = feature.matmul(self.fc_bin[0])
        feature = feature.permute(1, 0, 2).contiguous()

        return feature, None
