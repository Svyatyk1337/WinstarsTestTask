import torch
import torch.nn as nn


class block(nn.Module):
    """
    A single residual block in ResNet.

    Args:
        in_channels (int): Number of input channels.
        intermediate_channels (int): Number of intermediate channels.
        identity_downsample (nn.Module, optional): Downsampling layer for the identity connection. Defaults to None.
        stride (int, optional): Stride for the convolutional layers. Defaults to 1.
    """

    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4  # Expansion factor for the bottleneck
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        """Forward pass of the block."""
        identity = x.clone()  # Save input for identity connection

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(
                identity)  # Apply downsampling if needed

        x += identity  # Add identity connection
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    """
    ResNet model.

    Args:
        block (nn.Module): The residual block class to use.
        layers (list): Number of blocks in each layer.
        image_channels (int): Number of input image channels.
        num_classes (int): Number of output classes.
    """

    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64  # Initial number of channels
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(512 * 4, num_classes)  # Fully connected layer

    def forward(self, x):
        """Forward pass of the ResNet model."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        """Creates a layer of residual blocks."""
        identity_downsample = None
        layers = []

        # Downsampling for the first block in the layer if stride is not 1 or channels change
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels,
                  identity_downsample, stride)
        )

        # Update in_channels for subsequent blocks
        self.in_channels = intermediate_channels * 4

        # Remaining blocks in the layer (no downsampling needed)
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=10):
    """Creates a ResNet50 model."""
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)
