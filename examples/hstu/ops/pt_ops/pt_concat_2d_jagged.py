# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import Tuple


def concat_2D_jagged(values_a: torch.Tensor,
                     offsets_a: torch.Tensor,
                     values_b: torch.Tensor,
                     offsets_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch 原生 2D 不规则张量拼接算子
    """
    B = offsets_a.shape[0] - 1
    D = values_a.shape[1]
    out_values = []
    out_offsets = [0]
    for i in range(B):
        a_start, a_end = offsets_a[i].item(), offsets_a[i + 1].item()
        b_start, b_end = offsets_b[i].item(), offsets_b[i + 1].item()
        a_slice = values_a[a_start:a_end]
        b_slice = values_b[b_start:b_end]
        out_slice = torch.cat([a_slice, b_slice], dim=0)
        out_values.append(out_slice)
        out_offsets.append(out_offsets[-1] + out_slice.shape[0])
    out_values = torch.cat(out_values, dim=0)
    out_offsets = torch.tensor(out_offsets, device=values_a.device, dtype=offsets_a.dtype)
    return out_values, out_offsets
