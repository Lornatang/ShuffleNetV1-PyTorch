# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any

import torch
from torch import Tensor
from torch import nn

__all__ = [
    "SqueezeNet",
    "squeezenet",
]


class SqueezeNet(nn.Module):

    def __init__(
            self,
            dropout: float = 0.5,
            num_classes: int = 1000,
    ) -> None:
        super(SqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, (7, 7), (2, 2), (0, 0)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True),
            Fire(512, 64, 256, 256),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(512, num_classes, (1, 1), (1, 1), (0, 0)),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)

        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.classifier(out)
        out = torch.flatten(out, 1)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)


class Fire(nn.Module):
    def __init__(
            self,
            in_channels: int,
            squeeze_channels: int,
            expand1x1_channels: int,
            expand3x3_channels: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, (1, 1), (1, 1), (0, 0))
        self.squeeze_activation = nn.ReLU(True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, (1, 1), (1, 1), (0, 0))
        self.expand1x1_activation = nn.ReLU(True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, (3, 3), (1, 1), (1, 1))
        self.expand3x3_activation = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.squeeze(x)
        out = self.squeeze_activation(out)

        expand1x1 = self.expand1x1(out)
        expand1x1 = self.expand1x1_activation(expand1x1)
        expand3x3 = self.expand3x3(out)
        expand3x3 = self.expand3x3_activation(expand3x3)

        out = torch.cat([expand1x1, expand3x3], 1)

        return out


def squeezenet(**kwargs: Any) -> SqueezeNet:
    model = SqueezeNet(**kwargs)

    return model
