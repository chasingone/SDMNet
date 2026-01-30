import torch.nn as nn
import torch
import cv2
import numpy as np
import torch.nn.functional as F
#cuda_name='cuda:2'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()
def getDCT(image):###################################adiation

    # 进行DCT变换
    dct_image = cv2.dct(np.float32(image))

    # 获取DCT变换后的图像的行和列数
    rows, cols = dct_image.shape

    # 定义中频部分的频率范围，您可以根据需要进行调整
    low_freq, mid_freq, high_frep = rows/32, rows/2, rows/2  # 这里示例选择了中频部分的频率范围
    #low_freq, mid_freq, high_frep = 16, 64, 128  # 这里示例选择了中频部分的频率范围

    # 创建一个与DCT变换后的图像大小相同的零矩阵
    low_frequency_map = np.zeros_like(dct_image, dtype=np.float32)
    mid_frequency_map1 = np.zeros_like(dct_image, dtype=np.float32)

    frequency_maps=[]

    # 根据中频范围将系数分配到中频部分
    for i in range(rows):
        for j in range(cols):
            # 计算DCT系数的频率
            frequency = np.sqrt(i ** 2 + j ** 2)

            # 将频率位于中频范围内的系数分配到中频部分

            if 0<= frequency <= low_freq :
                low_frequency_map[i, j] = 1  # 标记为低频部分

            if low_freq <= frequency <= mid_freq:
                mid_frequency_map1[i, j] = 1  # 标记为中频部分

    # 使用逆DCT变换还原中频部分的图像
    low_freq_image = cv2.idct(np.multiply(dct_image, low_frequency_map))
    mid_freq_image = cv2.idct(np.multiply(dct_image, mid_frequency_map1))

    return low_freq_image, mid_freq_image

def Dct_inTensor(tensor):###################################adiation
    cuda_namne=tensor.device
    tensor=tensor.cpu()
    tensor = tensor.detach().numpy()
    low_dct_tensor = np.zeros_like(tensor)
    mid_dct_tensor = np.zeros_like(tensor)

    # 遍历每个通道中的二维特征图并执行DCT变换
    for channel in range(tensor.shape[1]):
        for batch in range(tensor.shape[0]):
            # 获取当前通道的特征图
            feature_map = tensor[batch, channel, :, :]
            low_freq_image, mid_freq_image = getDCT(np.float32(feature_map))

            low_dct_tensor[batch, channel, :, :] = low_freq_image
            mid_dct_tensor[batch, channel, :, :] = mid_freq_image

    low_dct_tensor=torch.from_numpy(low_dct_tensor)
    low_dct_tensor =low_dct_tensor.to(cuda_namne)
    mid_dct_tensor=torch.from_numpy(mid_dct_tensor)
    mid_dct_tensor = mid_dct_tensor.to(cuda_namne)

    return low_dct_tensor,mid_dct_tensor

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.convT = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)

        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        #x=x.to(device)
        out = self.convT(x)

        # out1,out2=Dct_inTensor(out)
        # out = torch.cat([out, out1,out2], dim=1)
        # out=self.conv1(out)

        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)
def compute_midian_infeature(x):

    batch,channel,height,width=x.size()
    flattened_feature=x.view(batch,channel,-1)
    median_feature=torch.quantile(flattened_feature,0.5,dim=2,keepdim=True)
    media_out=median_feature.view(batch,channel,1,1)
    return media_out
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim*6 , self.dim*6),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim*6, self.dim*2),
                    nn.Sigmoid()
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        meidan=compute_midian_infeature(x).view(B, self.dim * 2)
        y = torch.cat((avg, max,meidan), dim=1) # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4) # 2 B C 1 1
        x11=x1 + channel_weights[1] * x2
        x22=x2 + channel_weights[0] * x1
        return x11,x22

class fusion_feature(nn.Module):
    def __init__(self, dim, reduction=1):
        super(fusion_feature, self).__init__()
        self.conv_mix = nn.Conv2d(dim, dim, kernel_size=3,stride=1, padding=1)
        self.norm = nn.BatchNorm2d(dim)
        self.activation = nn.ReLU()

    def forward(self, x1, x2):
    # 在通道维度上进行全局平均池化

        x3 = F.adaptive_avg_pool2d(x1, (1, 1))
        x4 = F.adaptive_avg_pool2d(x2, (1, 1))

        sort_number=x1.size(1)
        # 为x4的通道添加序号
        sorted_x3, indices3 = torch.sort(x3, dim=1, descending=True)
        top_indices3 = indices3[:, :sort_number//2]

        sorted_x4, indices4 = torch.sort(x4, dim=1, descending=True)
        top_indices4 = indices4[:, :sort_number//2]

        # 根据挑选的通道，从x3中选择相应的特征通道
        T3 = x1.gather(1, top_indices3.expand(-1, x1.size(1) // 2, x1.size(2), x1.size(3)))
        T4 = x2.gather(1, top_indices4.expand(-1, x1.size(1) // 2, x1.size(2), x1.size(3)))
        out = torch.cat((T3, T4), dim=1)
        #out =self.activation(self.norm(self.conv_mix(out)))

        return out

class SDMNet(nn.Module):
    def __init__(self, config,n_channels=9, n_classes=1,bilinear=True,img_size=512,vis=False):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # Question here
        in_channels = 64
        self.inc = ConvBatchNorm(3, in_channels)
        self.inc1 = ConvBatchNorm(6, in_channels)
        self.weight=ChannelWeights(in_channels)

        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.weight1=ChannelWeights(in_channels*2)

        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.weight2 = ChannelWeights(in_channels * 4)

        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.weight3 = ChannelWeights(in_channels * 8)

        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)
        self.fusion_feature=fusion_feature(in_channels*8)

        self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2)
        self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2)
        self.up2 = UpBlock(in_channels*4, in_channels, nb_Conv=2)
        self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2)

        self.outc4 = nn.Conv2d(in_channels*4, n_classes, kernel_size=(1,1))
        self.outc3 = nn.Conv2d(in_channels*2, n_classes, kernel_size=(1,1))
        self.outc2 = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))
        self.outc1 = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))
        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        # Question here
        x = x.float()

        x1D,x2D=Dct_inTensor(x)
        x1D=x1D.to(x.device)
        x2D=x2D.to(x.device)
        x_DCt = torch.cat([x1D, x2D], dim=1)

        x1 = self.inc(x)
        x11 = self.inc1(x_DCt)
        x1,x11=self.weight(x1,x11)

        x2 = self.down1(x1)
        x22 = self.down1(x11)
        x2, x22 = self.weight1(x2, x22)

        x3 = self.down2(x2)
        x33 = self.down2(x22)
        x3, x33 = self.weight2(x3, x33)

        x4 = self.down3(x3)
        x44 = self.down3(x33)
        x4, x44 = self.weight3(x4, x44)
        x4 = self.fusion_feature(x4, x44)

        x5 = self.down4(x4)

        x = self.up4(x5, x4)
        x_1 = x
        x = self.up3(x, x3)
        x_2 = x
        x = self.up2(x, x2)
        x_3= x
        x = self.up1(x, x1)
        shape = x1.size()[2:]

        x_1 = F.interpolate(x_1, size=shape, mode='bilinear')
        x_2 = F.interpolate(x_2, size=shape, mode='bilinear')
        x_3 = F.interpolate(x_3, size=shape, mode='bilinear')

        logits4 = self.last_activation(self.outc1(x))
        logits3 = self.last_activation(self.outc2(x_3))
        logits2 = self.last_activation(self.outc3(x_2))
        logits1 = self.last_activation(self.outc4(x_1))

        # if self.last_activation is not None:
        #     logits = self.last_activation(self.outc1(x))
        #     # print("111")
        # else:
        #     logits = self.outc(x)
            # print("222")
        # logits = self.outc(x) # if using BCEWithLogitsLoss
        # print(logits.size())
        return logits4, logits3, logits2, logits1


