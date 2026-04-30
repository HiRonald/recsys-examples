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


def add_position_embeddings(jagged, jagged_offsets, high_inds, max_seq_len, dense, scale=1.0):
    """
    PyTorch 原生实现为变长序列添加位置编码
    """
    L, D = jagged.shape
    B = high_inds.shape[0]
    out = torch.empty_like(jagged)
    for b in range(B):
        start = jagged_offsets[b].item()
        end = jagged_offsets[b + 1].item()
        seq_len = end - start

        pos_ids = torch.arange(seq_len, device=jagged.device) + high_inds[b].item()
        pos_emb = dense[pos_ids]
        out[start:end, :] = jagged[start:end, :] + pos_emb * scale
    return out
