# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
#from .vit_seg_modeling_resnet_skip import ResNetV2
from .vit_seg_modeling_resnet_skip_SWIN_FANet_CBAM_UI import ResNetV2
from archs import UNet #AShwini Added
import math #AShwini added

from blocks_UI import MixPool #AShwini Added for FA-Net


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

#Ashwini Added below code to add SWIN transformer---------------------------------------

from utilswin.module import Attention, PreNorm, FeedForward, CrossAttention
#from utilswin.checkpoint import load_checkpoint #AShwini hashed for mmcv error
import torch.utils.checkpoint as checkpoint
#from mmseg.utils import get_root_logger #AShwini hashed for mmcv error
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.autograd import Variable
import torch.nn.functional as F

groups = 32

class Mlp2(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features) #.cuda()
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features) #.cuda()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp2(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchRecover(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=dim//2, num_groups=groups),
            nn.ReLU(inplace=True)
        )


    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.permute(0, 1, 2) # B ,C, L
        x = x.reshape(B, C, H, W)
        x = self.up(x) # B, C//2, H, W

        x = x.reshape(B, C//2, -1)
        x = x.permute(0, 2, 1)

        #x = Variable(torch.randn(B, H * 2, W * 2, C // 2))

        return x

class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        #print(x.shape)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        #print(x.shape)
        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth, 
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 up=True):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.up = up

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            if self.up:
                Wh, Ww = (H + 1) // 2, (W + 1) // 2
            else:
                Wh, Ww = H * 2, W * 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module): #AShwini: Patch making layer

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        #print("Patch Size = ", patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        #print(x.shape)
        # #Saving Image x for Visualization
        # full_image_array = x.cpu().numpy()
        # # Ensure the data type is uint8 (0-255)
        # #full_image_array = (full_image_array * 255).astype(np.uint8)
        # full_image_array = full_image_array.astype(np.uint8)
        # print(full_image_array.shape)
        # # Create a Pillow image from the NumPy array
        # # image = Image.fromarray(full_image_array)
        # # # Save the image to a file
        # # image.save('full_image.png')

        # # Save the array as an image using OpenCV
        # cv2.imwrite("full_image.png", full_image_array)


        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        
        #print(x.shape)
        # #Saving Patch x for Visualization
        # patch_array = x.cpu().numpy()
        # # Ensure the data type is uint8 (0-255)
        # patch_array = (patch_array * 255).astype(np.uint8)
        # # Create a Pillow image from the NumPy array
        # patchimg = Image.fromarray(patch_array)
        # # Save the image to a file
        # patchimg.save('patch.png')

        return x

# class MultiEmbed(nn.Module):

#     def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
#         super().__init__()
#         patch_size = to_2tuple(patch_size)
#         self.patch_size = patch_size

#         self.in_chans = in_chans
#         self.embed_dim = embed_dim

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1)
#         self.bn = nn.BatchNorm2d(embed_dim)
#         self.maxPool = nn.MaxPool2d(kernel_size=patch_size, stride=patch_size)
#         if norm_layer is not None:
#             self.norm = norm_layer(embed_dim)
#         else:
#             self.norm = None

#     def forward(self, x):
#         """Forward function."""
#         # padding
#         _, _, H, W = x.size()
#         if W % self.patch_size[1] != 0:
#             x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
#         if H % self.patch_size[0] != 0:
#             x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

#         x = self.proj(x)  # B C Wh Ww
#         x = self.bn(x)
#         x = self.maxPool(x)
#         if self.norm is not None:
#             Wh, Ww = x.size(2), x.size(3)
#             x = x.flatten(2).transpose(1, 2)
#             x = self.norm(x)
#             x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

#         return x


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=256,   #Ashwini changed it from 224
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,    #Ashwini changed it to 96 from 128 as the image size is 256 and not 512
                 depths=[12], # 2, 18, 2],    AShwini changed it to use only one layer os SWIN transformer with 2 swin blocks
                 num_heads=[4], #, 8, 16, 32],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.5,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=[0], #Ashwini changed from (0,1,2,3)
                 frozen_stages=-1, # frozen_stages= -1 means no stages or parts of the model are frozen.
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        #self.ape = ape
        #self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # # split image into non-overlapping patches
        # self.patch_embed = PatchEmbed(
        #     patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)


        # # absolute position embedding
        # if self.ape:
        #     pretrain_img_size = to_2tuple(pretrain_img_size)
        #     patch_size = to_2tuple(patch_size)
        #     patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

        #     self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
        #     trunc_normal_(self.absolute_pos_embed, std=.02)
        
        #AShwini check th ebelow code part------------------------
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            print('i_layer = ', i_layer)
            print('depth = ', depths[i_layer])
            print('num_heads = ', num_heads[i_layer])
            self.layers.append(layer)


        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        print('num_features = ', num_features)
        


        # add a norm layer for each output
        for i_layer in out_indices:
            print('i_layer = ', i_layer)
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            #logger = get_root_logger() #AShwini hashed for mmcv error
            #load_checkpoint(self, pretrained, strict=False, logger=logger) #AShwini hashed for mmcv error
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        # x = self.patch_embed(x) #Ashwini: Images converted to patches

        Wh, Ww = int(math.sqrt(x.size(1))), int(math.sqrt(x.size(1)))
        # if self.ape:
        #     # interpolate the position embedding to the corresponding size
        #     absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
        #     x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        # else:
        #     x = x.flatten(2).transpose(1, 2)
        # x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww) #Applied swin transformer layer containing multiple swin transformer blocks. 

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        outs = outs[0]
        #print('Type of out = ', type(outs), ' and shape of out = ', outs.shape)
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()
 #------------------------------------------------ASHWINI ADDED SWIN Transformer----------------------------------------------------------------------

#-----------Ashwini Added SWIN Decoder & TIF Blocks below--------------------------------------------------------------

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=groups//2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Decoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(Decoder, self).__init__()
    self.up = up_conv(in_channels, out_channels)
    #self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
        #coorAtt(out_channels),
        nn.ReLU(inplace=True)
        )
  def forward(self, x1): # x2):
    x1 = self.up(x1)
    #x2 = self.att_block(x1, x2) 
    #x1 = torch.cat((x2, x1), dim=1) #Ashwini hashed this as no skip channels to concatenate
    x1 = self.conv_relu(x1)
    return x1


class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch,num_groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch,num_groups=groups),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class Conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(Conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch,num_groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch,num_groups=groups),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class SwinUp(nn.Module):
    def __init__(self, dim):
        super(SwinUp, self).__init__()
        self.up = nn.Linear(dim, dim*2) #.cuda()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.norm(x)
        x = self.up(x)
        x = x.reshape(B, H, W, 2 * C)

        x0 = x[:,:,:,0:C//2]
        x1 = x[:,:,:,C//2:C]
        x2 = x[:,:,:,C:C+C//2]
        x3 = x[:,:,:,C+C//2:C*2]

        x0 = torch.cat((x0, x1), dim=1)
        x3 = torch.cat((x2, x3), dim=1)
        x = torch.cat((x0, x3), dim=2)

        #x = Variable(torch.randn(B, H * 2, W * 2, C // 2))

        x = x.reshape(B, -1, C // 2)
        return x



class SwinDecoder(nn.Module):

    def __init__(self,
                 embed_dim,
                 patch_size=4,
                 depths=2,
                 num_heads=6,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False):
        super(SwinDecoder, self).__init__()

        self.patch_norm = patch_norm

        # split image into non-overlapping patches

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        # build layers
        self.layer = BasicLayer(
            dim=embed_dim, #This will be the output channels from this layer
            depth=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint)
        
        self.up = up_conv(embed_dim, embed_dim) #AShwini changed out_channel from embed_dim//2 to embed_dim 
        self.conv_relu = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//2, kernel_size=1, stride=1, padding=0), # Ashwini changed from (embed_dim//2, embed_dim//4, ... )
            nn.ReLU()
        )


    def forward(self, x):
        """Forward function."""

        #print(x.shape)
        #for i in range(len(e_o)):
        #    layer = self.layers[i]
        #    x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
        #return x
        
        identity = x
        #print('x.shape = ', x.shape)
        B, C, H, W = x.shape
        #x = self.up(x) # B , C, 2H, 2W
        #print('x.shape2 = ', x.shape)
        x = x.reshape(B , C, H*W) #AShwini changed from C//2
        x = x.permute(0, 2, 1)

        x_out, H, W, x, Wh, Ww = self.layer(x, H, W) #Ashwini changed from H*2, W*2

        x = x.permute(0, 2, 1)
        x = x.reshape(B , C, H, W) #Ashwini changed from C//2
        # B, C//4 2H, 2W
        x = self.conv_relu(x) #Ashwini hashed it so that output dimensions are not halved further

        return x

class Swin_Decoder(nn.Module):
  def __init__(self, in_channels, depths, num_heads):
    super(Swin_Decoder, self).__init__()
    self.up = SwinDecoder(in_channels, depths=depths, num_heads=num_heads) #This will half the channels & upsample by a factor of 2
    #self.up1 = nn.Upsample(scale_factor=2)
    #self.up2 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1, bias=True)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(in_channels//2, in_channels//4, kernel_size=3, padding=1), #Ashwini multiplied input channels by a factor of 2 and kept the out channels as it is
        nn.ReLU(inplace=True)
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0), #Ashwini multiplied both in & out channels by a factor of 2
        nn.ReLU()
    )

  def forward(self, x1):
    #print('Shape of x1 (before SWIN Decode Block) = ', x1.shape)
    x1 = self.up(x1)
    #print('Shape of x1 (after SWIN Decode Block) = ', x1.shape)
    #x1 = self.up2(x1)
    #x2 = self.att_block(x1, x2)
    #print('Shape of x2 (skip) before conv2 = ', x2.shape)
    #x2 = self.conv2(x2)
    #print('Shape of x2 (skip) after conv2 = ', x2.shape)
    #x1 = torch.cat((x2, x1), dim=1)
    #print('Shape of x1 + x2 = ', x1.shape)
    out = self.conv_relu(x1)
    #print('Shape of output after whole SWIN Decoder Layer = ', out.shape)
    return out

# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
#             ]))
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x


# class Cross_Att(nn.Module):
#     def __init__(self, dim_s, dim_l):
#         super().__init__()
#         self.transformer_s = Transformer(dim=dim_s, depth=1, heads=3, dim_head=32, mlp_dim=128)
#         self.transformer_l = Transformer(dim=dim_l, depth=1, heads=1, dim_head=64, mlp_dim=256)
#         self.norm_s = nn.LayerNorm(dim_s)
#         self.norm_l = nn.LayerNorm(dim_l)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.linear_s = nn.Linear(dim_s, dim_l)
#         self.linear_l = nn.Linear(dim_l, dim_s)

#     def forward(self, e, r):
#        b_e, c_e, h_e, w_e = e.shape
#        e = e.reshape(b_e, c_e, -1).permute(0, 2, 1)
#        b_r, c_r, h_r, w_r = r.shape
#        r = r.reshape(b_r, c_r, -1).permute(0, 2, 1)
#        e_t = torch.flatten(self.avgpool(self.norm_l(e).transpose(1,2)), 1)
#        r_t = torch.flatten(self.avgpool(self.norm_s(r).transpose(1,2)), 1)
#        e_t = self.linear_l(e_t).unsqueeze(1)
#        r_t = self.linear_s(r_t).unsqueeze(1)
#        r = self.transformer_s(torch.cat([e_t, r],dim=1))[:, 1:, :]
#        e = self.transformer_l(torch.cat([r_t, e],dim=1))[:, 1:, :]
#        e = e.permute(0, 2, 1).reshape(b_e, c_e, h_e, w_e) 
#        r = r.permute(0, 2, 1).reshape(b_r, c_r, h_r, w_r) 
#        return e, r
    
    #-----------------------------------SWIN DEcoder Blocks above--------------------------------------------------------

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        #self.edge = None
        self.config = config
        img_size = _pair(img_size)
        #print('config.patches.get("grid") = ', config.patches.get("grid")) # came out to be (16,16)
        #print('img_size = ', img_size) #came out to be 256x256
        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16) #Ashwini: This gives the real patch size in input images (256x256 images in this case)
            #manual_patch_size = (4,4) # to manually select patch size and have to manually set n_patches accordingly 
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1]) # Ashwini will hash this to manually select n_patches in below code line
            #n_patches = 16
            #print('n_patches = ', n_patches) 
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor) #Ashwini hashed it
            #self.hybrid_model = UNet(1, in_channels) # Ashwini Added this
            in_channels = self.hybrid_model.width * 16    #64*16 # self.hybrid_model.width * 16 #Ashwini changed it
        #print('config.hidden_size = ', config.hidden_size) =====> this came out to be 768
        #print('patch_size = ', patch_size)
        #print('patch_size_real = ', patch_size_real)
        #print('manual patch_size = ', manual_patch_size)

        #From below given code you can play with patch sizes. Every patch is represented by a number using a convolution filter.
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels= 768, #config.hidden_size, This had to be changed as we are selecting SWIN output channels as 1024
                                       kernel_size=patch_size, #changed it from patch_size which was (1,1) to patch_size_real
                                       stride=patch_size)  #changed it from patch_size which was (1,1) to patch_size_real  
        pe = self.patch_embeddings
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, 768)) # config.hidden_size)) This also has to change to mtch above patch embeddings dimensions.

        self.dropout = Dropout(config.transformer["dropout_rate"])

        # ---- Adding edge attention branch ----
        self.edge_conv1 = BasicConv2d(256, 64, kernel_size=1)
        self.edge_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.edge_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.edge_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)


    def forward(self, x, m):
        if self.hybrid:
            x, features, swin_outs = self.hybrid_model(x, m)
        else:
            features = None
        #print('x shape before patch making = ', x.shape)
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        #print('x shape after patching = ', x.shape)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        #print('x shape after flattening & transposing = ', x.shape)
 
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        #print('Final embeddings shape = ', embeddings.shape)

        #print('Features[1] shape = ', features[1].shape)

         # ---- Adding edge guidance ----
        e1 = self.edge_conv1(features[1])
        e2 = self.edge_conv2(e1)
        edge_guidance = self.edge_conv3(e2)  # torch.Size([1, 64, 88, 88])
        lateral_edge = self.edge_conv4(edge_guidance)   # NOTES: Sup-2 (bs, 1, 88, 88) -> (bs, 1, 352, 352)
        lateral_edge = F.interpolate(lateral_edge,
                                     scale_factor=4,
                                     mode='bilinear')
        #print('edge map shape = ', lateral_edge.shape)

        return embeddings, features, lateral_edge #Ashwini added , lateral_edge

# Ashwini hashed below given classes (Block & Encoder) as they have been replace with SWIN Transformer Blocks--------------------------------

# class Block(nn.Module):
#     def __init__(self, config, vis):
#         super(Block, self).__init__()
#         self.hidden_size = config.hidden_size
#         self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         self.ffn = Mlp(config)
#         self.attn = Attention(config, vis)

#     def forward(self, x):
#         h = x
#         x = self.attention_norm(x)
#         x, weights = self.attn(x)
#         x = x + h

#         h = x
#         x = self.ffn_norm(x)
#         x = self.ffn(x)
#         x = x + h
#         return x, weights

#     def load_from(self, weights, n_block):
#         ROOT = f"Transformer/encoderblock_{n_block}"
#         with torch.no_grad():
#             query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
#             key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
#             value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
#             out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

#             query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
#             key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
#             value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
#             out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

#             self.attn.query.weight.copy_(query_weight)
#             self.attn.key.weight.copy_(key_weight)
#             self.attn.value.weight.copy_(value_weight)
#             self.attn.out.weight.copy_(out_weight)
#             self.attn.query.bias.copy_(query_bias)
#             self.attn.key.bias.copy_(key_bias)
#             self.attn.value.bias.copy_(value_bias)
#             self.attn.out.bias.copy_(out_bias)

#             mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
#             mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
#             mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
#             mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

#             self.ffn.fc1.weight.copy_(mlp_weight_0)
#             self.ffn.fc2.weight.copy_(mlp_weight_1)
#             self.ffn.fc1.bias.copy_(mlp_bias_0)
#             self.ffn.fc2.bias.copy_(mlp_bias_1)

#             self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
#             self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
#             self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
#             self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


# class Encoder(nn.Module):
#     def __init__(self, config, vis):
#         super(Encoder, self).__init__()
#         self.vis = vis
#         self.layer = nn.ModuleList()
#         self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         print('Number of transformer layers ', config.transformer["num_layers"])
#         for _ in range(config.transformer["num_layers"]):
#             layer = Block(config, vis)
#             self.layer.append(copy.deepcopy(layer))

#     def forward(self, hidden_states):
#         attn_weights = []
#         for layer_block in self.layer:
#             hidden_states, weights = layer_block(hidden_states)
#             if self.vis:
#                 attn_weights.append(weights)
#         encoded = self.encoder_norm(hidden_states)
#         return encoded, attn_weights

#-------------------------------------------------------------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        #self.encoder = Encoder(config, vis) #This has to be changed to encorporate SWIN Transformer
        self.encoder = SwinTransformer(depths=[12], num_heads=[4], drop_path_rate=0.5, patch_size=16, embed_dim=768) #Ashwini Added and changed from depths = [2,2,18,2] 
        #self.encoder.init_weights('checkpoints/swin_base_patch4_window7_224_22k.pth')
    def forward(self, input_ids, m):
        embedding_output, features, lateral_edge = self.embeddings(input_ids, m) # Ashwini added , lateral_edge
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)
        #print('shape of SWIN Transformer output = ', encoded.shape) # (B, n_patch, hidden) = (16, 256, 768)
        return encoded, features, lateral_edge


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels, #Ashwini changed it from =3
            use_batchnorm=True,
    ):
        super().__init__()
        #self.conv = VGGBlock(in_channels + skip_channels, out_channels, out_channels)
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        #self.p1 = MixPool(out_channels, out_channels) #Ashwini Added this for FANet

    def forward(self, x, skip): # Ashwini Changed from skip = None
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        #print('The shape of x = ', x.shape)
        x = self.conv1(x)
        #print('The shape of x after conv = ', x.shape)
        x = self.conv2(x)
        #print('The shape of x after conv 2 = ', x.shape)
        #x = self.p1(x, m)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1): #Ashwini Changed upsampling = 2 to increase size of images from 128x128 to 256x256
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity() #This means upsampling won't happen. 
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512 #AShwini changed from 512
        self.conv_more = Conv2dReLU( #Need to be done before giving to decoder block because transformer stage produces linear data vector. 
            768, #config.hidden_size, #768 originally, changed by Ashwini as SWIN Transformer output channels changed. 
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        #decoder_channels = (256, 128, 64, 16) #Ashwini changed it to add a SWIN layer at the end of CNN decoder channel before segmentation head
        #print('decoder channels = ', decoder_channels)
        in_channels = [head_channels] + list(decoder_channels[:-1]) #This will add head_channels value at start of decoder_channels & remove its last value
        #print('in_channels = ', in_channels)
        #decoder_channels[:-1] takes a slice of the decoder_channels list, excluding the last element ([:-1])
        #print('in_channels = ', str(in_channels))
        out_channels = decoder_channels # = (512,256,128,64)

        #AShwini changed below to make skip channels = 0 to 4
        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            #skip_channels = [512, 256, 128, 64]
            #Below loop is to make skip_channels as 0 if not applicable
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0] #Ashwini changed it to make skip channels as 0
        #skip_channels = [512, 256, 0, 64] #Ashwini Manually defining skip channels and overriding the above operations
        #print('n_skip = ', config.n_skip)
        #print('in_channels = ', in_channels)
        #print('out_channels = ', out_channels)
        #print('skip_channels = ', skip_channels)
        
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ] #Ashwini HAshed it for using SWIN Decoder Blocks
        self.blocks = nn.ModuleList(blocks) #Ashwini HAshed it for using SWIN Decoder Blocks
        self.p1 = MixPool(16, 16) #Ashwini added this for MixPool before Swin Transformer
        #self.p1 = MixPool(4, 4) #Ashwini added this for MixPool after Swin Transformer
        #Ashwini Added Below code part to add SWIN DECODER---------------------------------
        self.layer1 = Swin_Decoder(16, 4, 4) # (in_channels, depths, num_heads)
        #self.layer2 = Swin_Decoder(256, 2, 4)
        #self.layer3 = Swin_Decoder(128, 2, 2)
        #self.layer4 = Decoder(64, 32, 32)
        #self.layer5 = Decoder(64, 32, 32) # This block is expendable and segmentation-head can be directly applied by changing its in_channels to 64
        # self.layer5 = Decoder(64, 32, 32) # This block is expendable and segmentation-head can be directly applied by changing its in_channels to 32
        #-----------------------------------------------------------------------------------

    def forward(self, hidden_states, features, m): # Ashwini changed from features = None
        
        #Ashwini: Below given shape changing part is not required as already done in SwinTransformer Class but not done in normal Transformer-

        # B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden) #VERY IMPORTANT STAGE
        # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch)) # n_patches = 256, (h,w) = (16,16)
        # x = hidden_states.permute(0, 2, 1) #earrange the dimensions from (0,1,2) to (0,2,1)
        # x = x.contiguous().view(B, hidden, h, w) # gives (16, 768, 16,16)
        #-----------------------------------------------------------------------------------------------------------
        
        x = self.conv_more(hidden_states) # gives (16, 512, 16,16) 
        #Ashwini Hashed Below to replace it with SWIN decoder blocks----------------------------------X]
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                #features[2] = None #Ashwini Intentionally making 3rd skip connection from bottom as zero. 
                skip = features[i] if (i < self.config.n_skip) else None #Here the skip channels are defined
                #print('Skip connection shape = ', skip.shape)
            else:
                skip = None
            x = decoder_block(x, skip=skip) #Add m here if you are using MixPool feedback in the decoder_block
        #-----------------------------------------------------------------------------------------------X]
        x = self.p1(x, m) #Ashwini Added it before the Swin Transformer Layer. 
        x = self.layer1(x)
        #x = self.p1(x, m) #Ashwini Added it after the Swin Transformer Layer. 
        # d2 = self.layer2(d1, features[1])
        # d3 = self.layer3(d2, features[2])
        #d4 = self.layer4(d1)
        #d5 = self.layer5(d4)
        #x = d1
        #print('Shape of DecoderCup output = ', x.shape)
        return x


class VisionTransformerSWIN(nn.Module):
    def __init__(self, config, img_size=256, num_classes=1, zero_head=False, vis=False): #Ashwini changed num_classes from 21843
        super(VisionTransformerSWIN, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels= 5, #config['decoder_channels'][-1], #this will retrieve the last element from the list 'decoder_channels'
            out_channels=config['n_classes'],
            kernel_size=3, #Ashwini change it from 3
        )
        self.config = config

        #Ashwini Adding below part to add deep-supervision for additional loss calculation------------------
        self.loss1 = nn.Sequential(
                nn.Conv2d(768, 1, kernel_size=1, stride=1, padding=0), #As in_channels from the encoder = 768 
                nn.ReLU(),
                nn.Upsample(scale_factor=16) #As 16X16 image has to become 256X256
        )

        self.loss2 = nn.Sequential(
                nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Upsample(scale_factor=1)
        )
        #--------------------------------------------------------------------------------------------------------

    def forward(self, x, m):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, features, lateral_edge = self.transformer(x, m)  # (B, n_patch, hidden)  #Ashwini added , lateral_edge
        loss1 = self.loss1(x) # Ashwini added this Output from the encoder for Deep Supervision
        x = self.decoder(x, features, m)
        loss2 = self.loss2(x) #Ashwini added this Output from the Decoder for Deep Supervision
        #print ('2nd last output shape = ', x.shape)
        #print('Shape of the train_mask = ', m.shape)
        x = torch.cat([x, m], dim=1)
        logits = self.segmentation_head(x)
        return logits, lateral_edge, loss1, loss2 #Ashwini added , lateral_edge, loss1, loss2 

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            #Ashwini hashed below two line codes because Standard Transformer is replaced with SWIN Transformer
            #self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            #self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            #Ashwini hashed below three line codes because Standard Transformer is replaced with SWIN Transformer
            # Encoder whole
            # for bname, block in self.transformer.encoder.named_children():
            #     for uname, unit in block.named_children():
            #         unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


