from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum
from typing import Optional

import torch
import torch.nn as nn

from . import layers
from torch.nn import functional as F


class ModelType(Enum):
    CLASSIFICATION = 0
    SEGMENTATION = 1


class OutputType(Enum):
    DEFAULT = 0
    MULTISCALE = 1


class BrtResnetPyramidLite12(torch.nn.Module):
    """
    Network defining a Pixelseg BRT Residual Pyramid Network.
    A lite version of full BRT segmentation model, with disabled conv11 and the last upsampling layer moved to the end.
    The number of channels of the last upsampling layer will get reduced and thus the compute time is reduced as well.
    """

    def __init__(self, params: dict):
        """
        param params: WorkFlowConfig params.
        """
        super().__init__()
        self.modelType = ModelType.SEGMENTATION
        self.outputType = params.get("output_type", OutputType.DEFAULT)
        track_running_stats = True
        in_channels = params["input_dims"]
        self.num_classes = params["num_classes"]
        self.model_params = params["model_params"]
        num_block_layers = self.model_params["num_block_layers"]
        widening_factor = int(self.model_params["widening_factor"])
        upsample_mode = self.model_params["upsample_mode"]
        activation_fn = F.relu
        self.bias = self.model_params.get("bias", True)
        self.half_res_output = params.get("half_res_output", False)

        # params for tracking
        self.channels = [int(widening_factor * 32), int(widening_factor * 32), int(widening_factor * 64), int(widening_factor * 128), int(widening_factor * 128)]
        if params.get('pre_img', False):
            self.pre_img_layer = nn.Sequential(
                nn.Conv2d(in_channels, int(widening_factor * 32), kernel_size=5, stride=2,
                        padding=2, bias=False),
                nn.BatchNorm2d(int(widening_factor * 32)),
                nn.ReLU(inplace=True))
        if params.get('pre_hm', False):
            self.pre_hm_layer = nn.Sequential(
                nn.Conv2d(1, int(widening_factor * 32), kernel_size=5, stride=2,
                        padding=2, bias=False),
                nn.BatchNorm2d(int(widening_factor * 32)),
                nn.ReLU(inplace=True))

        self.conv1 = layers.ConvBatchNormBlock(
            in_channels=in_channels,
            out_channels=int(widening_factor * 32),
            activation_fn=layers.null_activation_fn,
            is_track_running_stats=track_running_stats,
            kernel_size=5,
            stride=2,
            padding=2,
            name="conv_bn_block_1",
            bias=self.bias,
        )

        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)

        self.res_block_2a = layers.ResidualBlockOriginal(
            num_block_layers=num_block_layers,
            in_channels=int(widening_factor * 32),
            filters=[int(widening_factor * 32), int(widening_factor * 32)],
            activation_fn=activation_fn,
            kernel_sizes=[3, 3],
            strides=[1, 1],
            dilation_rates=[1, 1],
            paddings=[1, 1],
            is_track_running_stats=track_running_stats,
            name="res_block_2a",
            bias=self.bias,
        )

        self.res_block_2b = layers.ResidualBlockOriginal(
            num_block_layers=num_block_layers,
            in_channels=int(widening_factor * 32),
            filters=[int(widening_factor * 32), int(widening_factor * 32)],
            activation_fn=activation_fn,
            kernel_sizes=[3, 3],
            strides=[1, 1],
            dilation_rates=[1, 1],
            paddings=[1, 1],
            is_track_running_stats=track_running_stats,
            name="res_block_2b",
            bias=self.bias,
        )

        self.res_block_3a = layers.ResidualBlockOriginal(
            num_block_layers=num_block_layers,
            in_channels=int(widening_factor * 32),
            filters=[int(widening_factor * 64), int(widening_factor * 64)],
            activation_fn=activation_fn,
            kernel_sizes=[5, 3],
            strides=[2, 1],
            dilation_rates=[1, 1],
            paddings=[2, 1],
            skip_conv_kernel_size=5,
            skip_conv_stride=2,
            skip_conv_dilation=1,
            skip_conv_padding=2,
            is_track_running_stats=track_running_stats,
            name="res_block_3a",
            bias=self.bias,
        )

        self.res_block_4a = layers.ResidualBlockOriginal(
            num_block_layers=num_block_layers,
            in_channels=int(widening_factor * 64),
            filters=[int(widening_factor * 128), int(widening_factor * 128)],
            activation_fn=activation_fn,
            kernel_sizes=[5, 3],
            strides=[2, 1],
            dilation_rates=[1, 1],
            paddings=[2, 1],
            skip_conv_kernel_size=5,
            skip_conv_stride=2,
            skip_conv_dilation=1,
            skip_conv_padding=2,
            is_track_running_stats=track_running_stats,
            name="res_block_4a",
            bias=self.bias,
        )

        self.res_block_5a = layers.ResidualBlockOriginal(
            num_block_layers=num_block_layers,
            in_channels=int(widening_factor * 128),
            filters=[int(widening_factor * 128), int(widening_factor * 128)],
            activation_fn=activation_fn,
            kernel_sizes=[5, 3],
            strides=[2, 1],
            dilation_rates=[1, 1],
            paddings=[2, 1],
            skip_conv_kernel_size=5,
            skip_conv_stride=2,
            skip_conv_dilation=1,
            skip_conv_padding=2,
            is_track_running_stats=track_running_stats,
            name="res_block_5a",
            bias=self.bias,
        )

        self.unpool6 = layers.Interpolate(
            scale_factor=2, mode=upsample_mode, num_channels=(widening_factor * 128)
        )
        self.conv6 = torch.nn.Conv2d(
            in_channels=int(widening_factor * 128),
            out_channels=int(widening_factor * 128),
            kernel_size=1,
            stride=1,
            padding=0,
        )
        torch.nn.init.xavier_uniform_(self.conv6.weight)

        # ele_add_6 (256 conv6(res_block_4a) + unpool6(res_block_5a))
        self.conv_bn_6_7 = layers.ConvBatchNormBlock(
            in_channels=int(widening_factor * 128),
            out_channels=int(widening_factor * 64),
            is_track_running_stats=track_running_stats,
            kernel_size=3,
            stride=1,
            activation_fn=activation_fn,
            padding=1,
            name="conv_bn_6_7",
            bias=self.bias,
        )

        self.unpool7 = layers.Interpolate(
            scale_factor=2, mode=upsample_mode, num_channels=(widening_factor * 64)
        )
        self.conv7 = torch.nn.Conv2d(
            in_channels=int(widening_factor * 64),
            out_channels=int(widening_factor * 64),
            kernel_size=1,
            stride=1,
            padding=0,
        )
        torch.nn.init.xavier_uniform_(self.conv7.weight)

        # ele_add_7 (128 conv7(res_block_3a) + unpool7(conv_bn_6_7(ele_add_6))
        self.conv_bn_7_8 = layers.ConvBatchNormBlock(
            in_channels=int(widening_factor * 64),
            out_channels=int(widening_factor * 32),
            is_track_running_stats=track_running_stats,
            kernel_size=3,
            stride=1,
            activation_fn=activation_fn,
            padding=1,
            name="conv_bn_7_8",
            bias=self.bias,
        )

        self.unpool8 = layers.Interpolate(
            scale_factor=2, mode=upsample_mode, num_channels=(widening_factor * 32)
        )
        self.conv8 = torch.nn.Conv2d(
            in_channels=int(widening_factor * 32),
            out_channels=int(widening_factor * 32),
            kernel_size=1,
            stride=1,
            padding=0,
        )
        torch.nn.init.xavier_uniform_(self.conv8.weight)

        # ele_add_8 (64 conv8(res_block_2b) + unpool8(conv_bn_7_8(ele_add_8))
        self.conv_bn_8_9 = layers.ConvBatchNormBlock(
            in_channels=int(widening_factor * 32),
            out_channels=int(widening_factor * 32),
            is_track_running_stats=track_running_stats,
            kernel_size=3,
            stride=1,
            activation_fn=activation_fn,
            padding=1,
            name="conv_bn_8_9",
            bias=self.bias,
        )

        self.unpool9 = layers.Interpolate(
            scale_factor=2, mode=upsample_mode, num_channels=(widening_factor * 32)
        )
        # concat9 (64 + 64 conv1)
        self.conv9 = layers.ConvBatchNormBlock(
            in_channels=int(widening_factor * 64),
            out_channels=int(widening_factor * 32),
            is_track_running_stats=track_running_stats,
            kernel_size=3,
            stride=1,
            activation_fn=activation_fn,
            padding=1,
            name="conv_bn_block_9",
            bias=self.bias,
        )

        self.conv10 = layers.ConvBatchNormBlock(
            in_channels=int(widening_factor * 32),
            out_channels=int(widening_factor * 16),
            is_track_running_stats=track_running_stats,
            kernel_size=3,
            stride=1,
            activation_fn=activation_fn,
            padding=1,
            name="conv_bn_block_10",
            bias=self.bias,
        )

        self.conv12 = layers.ConvBatchNormBlock(
            in_channels=int(widening_factor * 16),
            out_channels=int(widening_factor * 8),
            activation_fn=activation_fn,
            is_track_running_stats=track_running_stats,
            kernel_size=7,
            stride=1,
            padding=3,
            name="conv_bn_block_12",
            bias=self.bias,
        )

        self.conv13 = torch.nn.Conv2d(
            in_channels=int(widening_factor * 8),
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        torch.nn.init.xavier_uniform_(self.conv13.weight)

        self.unpool_logits = layers.Interpolate(
            scale_factor=2, mode=upsample_mode, num_channels=self.num_classes
        )

        # layers for scale output
        if self.outputType == OutputType.MULTISCALE:
            self.res_2b_output = layers.ScaleOutput(
                "res_2b_output",
                (
                    int(widening_factor * 32),
                    int(widening_factor * 16),
                    int(widening_factor * 8),
                    self.num_classes,
                ),
                activation_fn,
                bias=self.bias,
            )
            self.res_3a_output = layers.ScaleOutput(
                "res_3a_output",
                (
                    int(widening_factor * 64),
                    int(widening_factor * 24),
                    int(widening_factor * 8),
                    self.num_classes,
                ),
                activation_fn,
                bias=self.bias,
            )
            self.res_4a_output = layers.ScaleOutput(
                "res_4a_output",
                (
                    int(widening_factor * 128),
                    int(widening_factor * 32),
                    int(widening_factor * 8),
                    self.num_classes,
                ),
                activation_fn,
                bias=self.bias,
            )
            self.res_5a_output = layers.ScaleOutput(
                "res_5a_output",
                (
                    int(widening_factor * 128),
                    int(widening_factor * 32),
                    int(widening_factor * 8),
                    self.num_classes,
                ),
                activation_fn,
                bias=self.bias,
            )


    def forward(self, x: torch.Tensor, pre_img=None, pre_hm=None) -> Optional[torch.Tensor]:
        # encoder
        conv1 = self.conv1(x)  # N, 32WF, H/2, W/2

        # merge pre_img and pre_hm features
        if pre_img is not None:
            conv1 = conv1 + self.pre_img_layer(pre_img)
        if pre_hm is not None:
            conv1 = conv1 + self.pre_hm_layer(pre_hm)

        pool1 = self.pool1(conv1)  # N, 32WF, H/4, W/4
        res_block_2a = self.res_block_2a(pool1)  # N, 32WF, H/4, W/4
        res_block_2b = self.res_block_2b(res_block_2a)  # N, 32WF, H/4, W/4
        res_block_3a = self.res_block_3a(res_block_2b)  # N, 64WF, H/8, W/8
        res_block_4a = self.res_block_4a(res_block_3a)  # N, 128WF, H/16, W/16
        res_block_5a = self.res_block_5a(res_block_4a)  # N, 128WF, H/32, W/32

        # scale output
        if self.outputType == OutputType.MULTISCALE:
            res_2b_logits = self.res_2b_output(res_block_2b)
            res_3a_logits = self.res_3a_output(res_block_3a)
            res_4a_logits = self.res_4a_output(res_block_4a)
            res_5a_logits = self.res_5a_output(res_block_5a)

        # final output
        unpool6 = self.unpool6(res_block_5a)  # N, 128WF, H/16, W/16
        conv6 = self.conv6(res_block_4a)  # N, 128WF, H/16, W/16
        ele_add_6 = unpool6 + conv6  # N, 128WF, H/16, W/16
        conv_bn_6_7 = self.conv_bn_6_7(ele_add_6)  # N, 64WF, H/16, W/16
        unpool7 = self.unpool7(conv_bn_6_7)  # N, 64WF, H/8, W/8
        conv7 = self.conv7(res_block_3a)  # N, 64WF, H/8, W/8
        ele_add_7 = unpool7 + conv7  # N, 64WF, H/8, W/8
        conv_bn_7_8 = self.conv_bn_7_8(ele_add_7)  # N, 32WF, H/8, W/8
        unpool8 = self.unpool8(conv_bn_7_8)  # N, 32WF, H/4, W/4
        conv8 = self.conv8(res_block_2b)  # N, 32WF, H/4, W/4
        ele_add_8 = unpool8 + conv8  # N, 32WF, H/4, W/4
        conv_bn_8_9 = self.conv_bn_8_9(ele_add_8)  # N, 32WF, H/4, W/4
        unpool9 = self.unpool9(conv_bn_8_9)  # N, 32WF, H/2, W/2
        concat9 = torch.cat((unpool9, conv1), dim=1)  # N, 64WF, H/2, W/2
        conv9 = self.conv9(concat9)  # N, 32WF, H/2, W/2
        conv10 = self.conv10(conv9)  # N, 16WF, H/2, W/2
        conv12 = self.conv12(conv10)  # N, 8WF, H/2, W/2
        conv13 = self.conv13(conv12)  # N, C, H/2, W/2

        if self.half_res_output:
            logits = conv13
        else:
            logits = self.unpool_logits(conv13)  # N, C, H, W

        return res_block_2a, res_block_2b, res_block_3a, res_block_4a, res_block_5a

