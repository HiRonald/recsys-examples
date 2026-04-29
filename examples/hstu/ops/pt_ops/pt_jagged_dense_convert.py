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


def jagged_to_padded_dense(values, offsets, max_lengths, padding_value) -> torch.Tensor:
    return torch.ops.mxrec.jagged_to_padded_dense(
        values=values,
        offsets=offsets,
        max_lengths=max(max_lengths),
        padding_value=padding_value,
    )

def dense_to_jagged(dense, offsets, total_L=None):
    if total_L is None:
        total_L = offsets[0][-1].item()

    out0, out1 = torch.ops.mxrec.dense_to_jagged(dense, offsets, total_L)
    return out0, out1


class JaggedToPaddedDense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, offsets, max_lengths, padding_value):
        ctx.save_for_backward(*offsets)
        ctx.total_L = values.shape[0]
        return jagged_to_padded_dense(values, offsets, max_lengths, padding_value)

    @staticmethod
    def backward(ctx, grad_output):
        offsets = list(ctx.saved_tensors)
        total_L = ctx.total_L
        grad_values, _ = dense_to_jagged(grad_output, offsets, total_L)
        return grad_values, None, None, None


class DenseToJagged(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dense, offsets, total_L=None):
        ctx.save_for_backward(*offsets)
        out0, out1 = dense_to_jagged(dense, offsets, total_L)
        ctx.dense_shape = dense.shape
        return out0, out1

    @staticmethod
    def backward(ctx, grad_out0, grad_out1):
        offsets = list(ctx.saved_tensors)
        max_len = ctx.dense_shape[1]
        grad_dense = jagged_to_padded_dense(grad_out0, offsets, [max_len], 0.0)
        return grad_dense, None, None


def jagged_to_padded_dense_wrapper(values, offsets, max_lengths, padding_value):
    return JaggedToPaddedDense.apply(values, offsets, max_lengths, padding_value)


def dense_to_jagged_wrapper(dense, offsets, total_L=None):
    return DenseToJagged.apply(dense, offsets, total_L)
