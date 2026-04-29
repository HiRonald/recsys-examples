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
from typing import Tuple

import torch


def split_2D_jagged(
        values: torch.Tensor,
        offsets_a: torch.Tensor,
        offsets_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    L, D = values.shape
    B = offsets_a.shape[0] - 1
    values_a_list = []
    values_b_list = []

    for i in range(B):
        a_start = offsets_a[i].item()
        a_end = offsets_a[i + 1].item()
        b_start = offsets_b[i].item()
        b_end = offsets_b[i + 1].item()
        values_a_list.append(values[a_start:a_end])
        values_b_list.append(values[b_start:b_end])
    values_a = torch.cat(values_a_list, dim=0) if values_a_list else torch.empty(0, D, device=values.device)
    values_b = torch.cat(values_b_list, dim=0) if values_b_list else torch.empty(0, D, device=values.device)
    return values_a, values_b
