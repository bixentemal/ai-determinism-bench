"""MLX-native ResNet-50, structured to mirror torchvision's module naming so the
pretrained torchvision weights port over mechanically (no extra dependency).

MLX convs are NHWC with weights laid out (O, H, W, I); torchvision convs are NCHW
with weights (O, I, H, W), so 4-D conv weights are transposed (0,2,3,1) on load.
BatchNorm runs in eval mode (running stats), matching inference.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _conv(in_c, out_c, k, stride=1, pad=0):
    return nn.Conv2d(in_c, out_c, kernel_size=k, stride=stride, padding=pad, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_c, planes, stride=1, downsample=False):
        super().__init__()
        out_c = planes * self.expansion
        self.conv1 = _conv(in_c, planes, 1)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = _conv(planes, planes, 3, stride=stride, pad=1)
        self.bn2 = nn.BatchNorm(planes)
        self.conv3 = _conv(planes, out_c, 1)
        self.bn3 = nn.BatchNorm(out_c)
        # nn.Sequential(conv, bn) -> keys downsample.0 / downsample.1 (matches torchvision)
        self.downsample = (
            [_conv(in_c, out_c, 1, stride=stride), nn.BatchNorm(out_c)] if downsample else None
        )

    def __call__(self, x):
        identity = x
        out = nn.relu(self.bn1(self.conv1(x)))
        out = nn.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample[1](self.downsample[0](x))
        return nn.relu(out + identity)


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = _conv(3, 64, 7, stride=2, pad=3)
        self.bn1 = nn.BatchNorm(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, in_c, planes, blocks, stride):
        layers = [Bottleneck(in_c, planes, stride=stride, downsample=True)]
        out_c = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_c, planes))
        return layers

    def __call__(self, x):  # x: NHWC
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                x = block(x)
        x = mx.mean(x, axis=(1, 2))  # global average pool over H, W
        return self.fc(x)


def load_pretrained() -> ResNet50:
    """Build the MLX model and load torchvision IMAGENET1K_V2 weights into it."""
    import torchvision

    model = ResNet50()
    model.eval()

    tv = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    ).eval()

    weights = []
    for key, tensor in tv.state_dict().items():
        if key.endswith("num_batches_tracked"):
            continue  # mlx BatchNorm has no such buffer
        arr = tensor.numpy()
        if arr.ndim == 4:  # conv weight: (O, I, H, W) -> (O, H, W, I)
            arr = arr.transpose(0, 2, 3, 1)
        weights.append((key, mx.array(arr)))

    model.load_weights(weights)
    mx.eval(model.parameters())
    return model
