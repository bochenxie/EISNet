import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


""" ---------------------------------- """
"""   Proposed modules in this paper   """

# Activity-Aware Event Integration Module (AEIM)
class AEIM(nn.Module):
    def __init__(self, in_dim=1, out_dim=32):
        super().__init__()
        self.in_dim = in_dim
        self.stem = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=7, stride=4, padding=3, bias=False),
                                  nn.BatchNorm2d(out_dim),
                                  nn.ReLU(inplace=True))
        self.pool1 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                                   nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_dim),
                                   nn.ReLU(inplace=True))
        self.pool2 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=4, padding=2),
                                   nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_dim),
                                   nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(out_dim),
                                    nn.ReLU(inplace=True))
        self.attn = SpatialAttention(kernel_size=7)

    def forward(self, ev, map):
        _, C, H, W = ev.shape
        map = rearrange(map, 'B (D C) H W -> (B D) C H W', C=self.in_dim)
        # Multi-scale activity features
        map1 = self.stem(map)
        _, _, H1, W1 = map1.shape
        map2 = self.pool1(map1)
        map2 = F.interpolate(map2, size=(H1, W1), mode='bilinear', align_corners=False)
        map3 = self.pool2(map1)
        map3 = F.interpolate(map3, size=(H1, W1), mode='bilinear', align_corners=False)
        # Feature fusion
        map = self.fusion(map1 + map2 + map3)
        # Activity map
        mask = self.attn(map)
        mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        mask = rearrange(mask, '(B D) C H W -> B (D C) H W', D=C)
        # Recalibrate per-channel frame
        ev = ev * mask + ev
        return ev


# Modality Recalibration and Fusion Module (MRFM)
class MRFM(nn.Module):
    def __init__(self, dim=[32, 64]):
        super().__init__()
        self.dim_ev, self.dim_img = dim
        # Recalibration block
        self.sigmoid = nn.Sigmoid()
        ## CA
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d((self.dim_ev + self.dim_img), (self.dim_ev + self.dim_img) // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d((self.dim_ev + self.dim_img) // 16, (self.dim_ev + self.dim_img), 1, bias=False))
        ## SA
        self.conv = nn.Conv2d(4, 2, 7, padding=7 // 2, bias=False)
        ## Control factors
        self.gamma_ev = nn.Parameter(0 * torch.ones(self.dim_ev, 1, 1), requires_grad=True)
        self.gamma_img = nn.Parameter(0 * torch.ones(self.dim_img, 1, 1), requires_grad=True)
        # Adaptive layer for event modality
        if self.dim_ev != self.dim_img:
            self.proj = nn.Conv2d(self.dim_ev, self.dim_img, kernel_size=1, bias=False)
        # Cross-attention block
        self.norm_ev = LayerNorm(normalized_shape=self.dim_img, data_format='channels_first')
        self.norm_img = LayerNorm(normalized_shape=self.dim_img, data_format='channels_first')
        self.i2e = EfficientCrossAttention(in_channels_x=self.dim_img, in_channels_y=self.dim_img, key_channels=self.dim_img,
                                           head_count=4, value_channels=self.dim_img)
        self.e2i = EfficientCrossAttention(in_channels_x=self.dim_img, in_channels_y=self.dim_img, key_channels=self.dim_img,
                                           head_count=4, value_channels=self.dim_img)
        # Gated fusion block
        self.gate = nn.Sequential(nn.Conv2d(self.dim_img * 2, self.dim_img * 2 // 16, kernel_size=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(self.dim_img * 2 // 16, 2, kernel_size=1, bias=False),
                                  nn.Softmax(dim=1))
        self.norm_ffn = LayerNorm(normalized_shape=self.dim_img, data_format='channels_first')
        self.ffn = FFN(in_features=self.dim_img, hidden_features=self.dim_img // 4)

    def forward(self, ev, img):
        B, D, H, W = ev.shape
        # Feature Recalibration
        ## Channel
        mm = torch.cat([ev, img], dim=1)
        ca = self.sigmoid(self.fc(self.avg_pool(mm)) + self.fc(self.max_pool(mm)))
        ev_rec, img_rec = ev * ca[:, :D, :, :], img * ca[:, D:, :, :]
        ## Spatial
        sa = torch.cat([torch.mean(ev_rec, dim=1, keepdim=True), torch.max(ev_rec, dim=1, keepdim=True)[0],
                        torch.mean(img_rec, dim=1, keepdim=True), torch.max(img_rec, dim=1, keepdim=True)[0]], dim=1)
        sa = self.sigmoid(self.conv(sa))
        ev_rec, img_rec = ev_rec * sa[:, 0:1, :, :] * self.gamma_ev + ev, img_rec * sa[:, 1:2, :, :] * self.gamma_img + img
        ev_out, img_out = ev_rec, img_rec
        # Feature interaction
        if self.dim_ev != self.dim_img:
            ev_rec_p = self.proj(ev_rec)
        else:
            ev_rec_p = ev_rec
        ev_f, img_f = self.norm_ev(ev_rec_p), self.norm_img(img_rec)
        ev_f = self.i2e(ev_f, img_f) + ev_rec_p
        img_f = self.e2i(img_f, ev_f) + img_rec
        # Gated feature fusion
        out = self.gate(torch.cat([ev_f, img_f], dim=1))
        out = ev_rec_p * out[:, 0:1, :, :] + img_rec * out[:, 1:2, :, :]
        out = self.ffn(self.norm_ffn(out)) + out
        return out, ev_out, img_out

""" ---------------------------------- """
""" ---------------------------------- """


# ---------------------------------------------------------------
# Some useful functions
# References:
# [efficient-attention] https://github.com/cmsflash/efficient-attention
# [CBAM.PyTorch] https://github.com/luuuyi/CBAM.PyTorch
# [SegFormer] https://github.com/NVlabs/SegFormer
# [ConvNeXt]
# ---------------------------------------------------------------

class EfficientCrossAttention(nn.Module):

    def __init__(self, in_channels_x, in_channels_y, key_channels, head_count, value_channels):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels_y, key_channels, 1)
        self.queries = nn.Conv2d(in_channels_x, key_channels, 1)
        self.values = nn.Conv2d(in_channels_y, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels_x, 1)

    def forward(self, x, y):
        n, _, h, w = x.size()
        keys = self.keys(y).reshape((n, self.key_channels, h * w))
        queries = self.queries(x).reshape(n, self.key_channels, h * w)
        values = self.values(y).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                            :,
                            i * head_key_channels: (i + 1) * head_key_channels,
                            :
                            ], dim=2)
            query = F.softmax(queries[
                              :,
                              i * head_key_channels: (i + 1) * head_key_channels,
                              :
                              ], dim=1)
            value = values[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                    ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                    context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU(inplace=True), drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)
        self.act = act_layer
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
