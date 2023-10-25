#!/usr/bin/env python3
import logging

from collections import OrderedDict
from functools import lru_cache
from typing import AnyStr, Callable, List, Optional, Tuple

import torch
from torch.nn import functional as F

BATCH_NORM_EPSILON = 1e-5


@lru_cache(None)
def warn_once(msg: AnyStr):
    """
    Function for throwing a warning message only once for that message
    :param msg: Warning message string
    :return:
    """
    logging.warning(msg)


class ConvBatchNormBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_track_running_stats: bool,
        kernel_size: int,
        stride: int,
        activation_fn: torch.nn.Module = None,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        bn_momentum: float = 0.1,
        bn_affine: bool = True,
        name: str = "",
    ):
        """
        https://pytorch.org/docs/stable/nn.html#convolution-layers
        https://pytorch.org/docs/stable/nn.html?highlight=batch%20norm#torch.nn.BatchNorm2d
        """
        super().__init__()

        self.name = name
        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.batch_norm_layer = torch.nn.BatchNorm2d(
            num_features=out_channels,
            eps=BATCH_NORM_EPSILON,
            momentum=bn_momentum,
            affine=bn_affine,
            track_running_stats=is_track_running_stats,
        )
        if activation_fn is None:
            self.activation_fn = null_activation_fn
        else:
            self.activation_fn = activation_fn
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        torch.nn.init.xavier_uniform_(self.conv_layer.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        conv_tensor = self.conv_layer(inputs)
        bn_tensor = self.batch_norm_layer(conv_tensor)
        return self.activation_fn(bn_tensor)


class ResidualBlockOriginal(torch.nn.Module):
    def __init__(
        self,
        num_block_layers: int,
        in_channels: int,
        filters: List[int],
        activation_fn: torch.nn.Module,
        kernel_sizes: List[int],
        strides: List[int],
        dilation_rates: List[int],
        paddings: List[int],
        skip_conv_kernel_size: Optional[int] = None,
        skip_conv_stride: Optional[int] = None,
        skip_conv_dilation: Optional[int] = None,
        skip_conv_padding: Optional[int] = None,
        is_track_running_stats: bool = True,
        name: str = "",
        bias: bool = False,
    ):

        super().__init__()
        self.name = name
        self.num_block_layers = num_block_layers
        self.filters = filters
        self.activation_fn = activation_fn

        # Conv params
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.dilation_rates = dilation_rates
        self.paddings = paddings

        if len(filters) != num_block_layers:
            raise ValueError("filters array must have num_layers elements.")
        if len(kernel_sizes) != num_block_layers:
            raise ValueError("kernel_sizes array must have num_layers elements.")
        if len(strides) != num_block_layers:
            raise ValueError("strides array must have num_layers elements.")
        if len(dilation_rates) != num_block_layers:
            raise ValueError("dilation_rates array must have num_layers elements.")
        if len(paddings) != num_block_layers:
            raise ValueError("paddings array must have num_layers elements.")

        layers_dict = OrderedDict()
        current_in_channels = in_channels
        for i in range(self.num_block_layers):
            layers_dict["conv_{}".format(i)] = torch.nn.Conv2d(
                in_channels=current_in_channels,
                out_channels=filters[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                dilation=dilation_rates[i],
                groups=1,
                bias=bias,
            )

            layers_dict["batch_norm_{}".format(i)] = torch.nn.BatchNorm2d(
                num_features=filters[i],
                eps=BATCH_NORM_EPSILON,
                momentum=0.1,
                affine=True,
                track_running_stats=is_track_running_stats,
            )
            current_in_channels = filters[i]

        if all(
            x is not None
            for x in [
                skip_conv_kernel_size,
                skip_conv_stride,
                skip_conv_dilation,
                skip_conv_padding,
            ]
        ):
            layers_dict["skip_connection_conv"] = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=filters[-1],
                kernel_size=skip_conv_kernel_size,
                stride=skip_conv_stride,
                padding=skip_conv_padding,
                dilation=skip_conv_dilation,
                bias=bias,
            )
            layers_dict["skip_connection_bn"] = torch.nn.BatchNorm2d(
                num_features=filters[-1],
                eps=BATCH_NORM_EPSILON,
                momentum=0.1,
                affine=True,
                track_running_stats=is_track_running_stats,
            )

        # Init layers in nn.Sequential wrapper with layer order preserved from OrderedDict.
        self.sequential_layers = torch.nn.Sequential(layers_dict)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for name, layer in self.sequential_layers.named_children():
            if "conv" in name:
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        identity_tensor = input_tensor
        residual_tensor = input_tensor
        for i in range(self.num_block_layers):
            if not (
                hasattr(self.sequential_layers, "conv_{}".format(i))
                and hasattr(self.sequential_layers, "batch_norm_{}".format(i))
            ):
                raise LookupError("Could not find conv and batch_norm layers")

            conv_layer = getattr(self.sequential_layers, "conv_{}".format(i))
            bn_layer = getattr(self.sequential_layers, "batch_norm_{}".format(i))

            residual_tensor = conv_layer(residual_tensor)
            residual_tensor = bn_layer(residual_tensor)

            # Do not attach a activation to last conv layer before residual connection.
            if i < (self.num_block_layers - 1):
                residual_tensor = self.activation_fn(residual_tensor)

        # Extra conv layer to increase input dimension to match with output dimension.
        if not (
            hasattr(self.sequential_layers, "skip_connection_conv")
            and hasattr(self.sequential_layers, "skip_connection_bn")
        ):
            eltwise_add_tensor = residual_tensor + identity_tensor
            return self.activation_fn(eltwise_add_tensor)
        else:
            skip_conv_layer = getattr(self.sequential_layers, "skip_connection_conv")
            skip_bn_layer = getattr(self.sequential_layers, "skip_connection_bn")

            skip_conv_tensor = skip_conv_layer(identity_tensor)
            skip_bn_tensor = skip_bn_layer(skip_conv_tensor)
            eltwise_add_tensor = residual_tensor + skip_bn_tensor
            return self.activation_fn(eltwise_add_tensor)


class Interpolate(torch.nn.Module):
    """
    For some reason torch doesn't have a class version of this functional:
    https://pytorch.org/docs/stable/nn.html#torch.nn.functional.interpolate
    """

    def __init__(
        self,
        size: Optional[Tuple[int, int]] = None,
        scale_factor: Optional[float] = None,
        mode: str = "nearest",
        align_corners: bool = True,
        num_channels: Optional[int] = None,
    ):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners if mode != "nearest" else None
        self.num_channels = num_channels
        if num_channels is not None and scale_factor == 2 and self.mode == "nearest":
            self.upsample = torch.nn.ConvTranspose2d(
                num_channels, num_channels, 2, stride=2, groups=1, bias=False
            )
            weight = torch.zeros(num_channels, num_channels, 2, 2)
            for i in range(num_channels):
                weight[i, i] = 1
            self.upsample.weight = torch.nn.parameter.Parameter(data=weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        IMPORTANT NOTE: Algorithm, Assumes that x -> NCHW
        scale_factor parameter usage is converted to destination size output because TensorRt doesn't
        handle floor Op that comes with the use of scale factor. Pytorch internally does a floor operation to
        go from source size to destination size with scale factor parameter that errors out in TensorRt
        engine creation side.
        """
        if self.scale_factor is not None:
            tensor_height, tensor_width = x.shape[-2:]
            output_size = (
                int(tensor_height * self.scale_factor),
                int(tensor_width * self.scale_factor),
            )
        elif self.size is not None:
            output_size = self.size
        else:
            raise ValueError("Either size or scale_factor must be specified")

        if not self.training and self.num_channels is not None and self.mode == "nearest":
            warn_once("Mimicking nearest upsampling (x2) using conv transpose")
            return self.upsample(x)

        return F.interpolate(x, size=output_size, mode=self.mode, align_corners=self.align_corners)


class ScaleOutput(torch.nn.Module):
    """
    Additional module for scaled outputs.
    It takes an intermediate feature map as input and outputs a same scaled segmentation map that
    has the desired number of classes. For example, if the input shape is N,C,H,W, the output will
    be N,C',H,W, with only the number of channels C' changed to the desired class number.
    The number of layers in this module depends on the param channels. Basically we need several
    3x3conv/BN/ReLU layers and a final 1x1conv layer. At least the 1x1conv layer is needed.
    Every two adjacent numbers in param channels determine a layer's in and out channels. For example,
    if channels = [c1, c2, c3, ..., cn], the [in, out] channels of each conv layer is [c1, c2], [c2, c3], etc.
    In this case, to match the channels of input intermediate feature map and output segmentation map,
    c1 should equal to C and cn should equal to C'.
    """

    def __init__(
        self,
        name: str,
        channels: Tuple,
        activation_fn: torch.nn.Module,
        is_track_running_stats: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.name = name
        assert len(channels) >= 2, "please enter at least two channel numbers for the scaled output"
        layers_dict = OrderedDict()
        for i in range(len(channels) - 2):
            layers_dict["conv_bn_block_" + str(i)] = ConvBatchNormBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                activation_fn=activation_fn,
                is_track_running_stats=is_track_running_stats,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            )
        layers_dict["conv"] = torch.nn.Conv2d(
            in_channels=channels[-2], out_channels=channels[-1], kernel_size=1, stride=1, padding=0
        )
        torch.nn.init.xavier_uniform_(layers_dict["conv"].weight)
        self.layers = torch.nn.Sequential(layers_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


"""
Activation functions:
"""


class Activation(torch.nn.Module):
    def __init__(self, activation_fn: Callable):
        super().__init__()
        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_fn(x)


def swish_fn(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    return x * torch.sigmoid(x * beta)


def null_activation_fn(x: torch.Tensor) -> torch.Tensor:
    return x


def make_one_hot(labels: torch.Tensor, num_classes: int, ignore_label: int) -> torch.Tensor:
    """
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    This function also makes the ignore label to be num_classes + 1 index and slices it from target.


    :labels : torch.autograd.Variable of torch.cuda.LongTensor N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    :param num_classes: Number of classes in labels.
    :return target : torch.autograd.Variable of torch.cuda.FloatTensor
    N x C x H x W, where C is class number. One-hot encoded.
    """
    labels = labels.long()
    labels = torch.where(labels >= num_classes, num_classes, labels)
    target = F.one_hot(labels.squeeze(1), num_classes + 1).permute(0, 3, 1, 2)
    target = target[:, :num_classes, :, :]
    return target

