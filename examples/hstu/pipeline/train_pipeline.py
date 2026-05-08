#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ============================================
# 章节1: 导入模块
# 本文件实现了训练流水线类，用于重叠数据加载、稀疏数据分发、前向/反向计算
# ============================================

# pyre-strict
import abc  # 抽象基类
import logging
import os
import time
from collections import deque  # 双端队列，用于批次缓存
from typing import (
    Any,
    Callable,
    Deque,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    cast,
)

import nvtx  # NVIDIA工具扩展，用于性能分析
import torch

# 导入分布式工具
from commons.utils.distributed_utils import collective_assert
from distributed.finalize_model_grads import finalize_model_grads
from megatron.core import parallel_state
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel

# 导入流水线工具函数和类
from pipeline.utils import (
    In,
    Out,
    PipelinedForward,
    PipelinedPostproc,
    PrefetchPipelinedForward,
    PrefetchTrainPipelineContext,
    TrainPipelineContext,
    _override_input_dist_forwards,
    _pipeline_detach_model,
    _prefetch_embeddings,
    _rewrite_model,
    _start_data_dist,
    _to_device,
    _wait_for_batch,
)

from torch.autograd.profiler import record_function
from torchrec.distributed.dist_data import KJTAllToAllTensorsAwaitable
from torchrec.distributed.model_parallel import ShardedModule
from torchrec.distributed.types import Awaitable
from torchrec.pt2.checks import is_torchdynamo_compiling
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# 日志记录器
logger: logging.Logger = logging.getLogger(__name__)

# 导入FBGEMM稀疏操作（非部署模式）
if not torch._running_with_deploy():
    torch.ops.import_module("fbgemm_gpu.sparse_ops")


# ============================================
# 章节2: 抽象基类定义
# 定义训练流水线的通用接口
# ============================================

class TrainPipeline(abc.ABC, Generic[In, Out]):
    """
    训练流水线的抽象基类。
    Abstract base class for training pipelines.
    
    所有具体的流水线实现都需要继承此类并实现progress方法。
    All concrete pipeline implementations must inherit from this class and implement the progress method.
    """
    @abc.abstractmethod
    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        """
        推进流水线一步，处理一个批次。
        Advance the pipeline by one step, processing a batch.
        
        参数/Args:
            dataloader_iter: 数据加载器迭代器 / dataloader iterator
            
        返回/Returns:
            Out: 当前批次的输出 / current batch output
        """
        pass


# ============================================
# 章节3: 标准稀疏分布式训练流水线
# TrainPipelineSparseDist - 最基础的流水线实现
# ============================================

class TrainPipelineSparseDist(TrainPipeline[In, Out]):
    """
    标准稀疏分布式训练流水线。
    Standard sparse distributed training pipeline.
    
    This pipeline overlaps device transfer, and `ShardedModule.input_dist()` with
    forward and backward. This helps hide the all2all latency while preserving the
    training forward / backward ordering.
    
    该流水线重叠以下三个阶段：
    - Stage 3: 前向/反向计算 - 使用默认CUDA流 / forward, backward - uses default CUDA stream
    - Stage 2: ShardedModule.input_dist() - 使用data_dist CUDA流
    - Stage 1: 设备数据传输 - 使用memcpy CUDA流 / device transfer - uses memcpy CUDA stream
    
    通过三阶段重叠，可以隐藏all2all通信延迟，同时保持训练顺序。
    Through three-stage overlapping, all2all communication latency can be hidden while preserving training order.
    
    参数/Args:
        model: 要流水的模型 / model to pipeline
        optimizer: 优化器 / optimizer to use
        device: 计算设备（cuda/npu）/ device for computation
        execute_all_batches: 是否执行所有批次（包括流水线中剩余的）/ executes remaining batches in pipeline
        apply_jit: 是否对非分片模块应用torch.jit.script / apply torch.jit.script to non-pipelined modules
    """
    
    # 在_rewrite_model中使用的PipelinedForward类
    _pipelined_forward_type = PipelinedForward

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        context_type: Type[TrainPipelineContext] = TrainPipelineContext,
        pipeline_postproc: bool = False,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
    ) -> None:
        """
        初始化训练流水线。
        """
        # 保存基本组件
        self._model = model
        self._optimizer = optimizer
        self._device = device
        self._execute_all_batches = execute_all_batches
        self._apply_jit = apply_jit

        # CUDA设备检查：Dynamo不支持CUDA流指定
        if device.type == "cuda":
            assert (
                not is_torchdynamo_compiling()
            ), "训练流水线依赖CUDA流，Dynamo不支持此特性"

        # 创建设备流上下文管理器
        self._stream_context = (
            torch.get_device_module(self._device).stream
            if self._device.type in ["cuda", "mtia", "npu"]
            else torch.cuda.stream
        )

        # 创建高优先级（-1）的memcpy流，用于设备间数据传输
        self._memcpy_stream: Optional[torch.Stream] = (
            (torch.get_device_module(device).Stream(priority=-1))
            if device.type in ["cuda", "mtia", "npu"]
            else None
        )
        
        # 创建高优先级（-1）的data_dist流，用于稀疏数据分发
        self._data_dist_stream: Optional[torch.Stream] = (
            (torch.get_device_module(device).Stream(priority=-1))
            if device.type in ["cuda", "mtia", "npu"]
            else None
        )

        # 用于保存原始forward函数和KJT分发函数的列表
        self._original_forwards: List[Callable[..., Any]] = []
        self._original_kjt_dist_forwards: List[
            Callable[[KeyedJaggedTensor], Awaitable[KJTAllToAllTensorsAwaitable]]
        ] = []

        # 流水线状态变量
        self._model_attached = True
        self._pipeline_postproc = pipeline_postproc

        # 批次和上下文管理
        self._next_index: int = 0  # 下一个批次的索引
        self.contexts: Deque[TrainPipelineContext] = deque()  # 上下文双端队列
        self._pipelined_modules: List[ShardedModule] = []  # 流水线模块列表
        self._pipelined_postprocs: List[PipelinedPostproc] = []  # 后处理模块列表
        self.batches: Deque[Optional[In]] = deque()  # 批次双端队列
        self._dataloader_iter: Optional[Iterator[In]] = None
        self._dataloader_exhausted: bool = False
        self._context_type: Type[TrainPipelineContext] = context_type

        # 模型前向函数（可以是自定义的或默认的model）
        self._model_fwd: Callable[[Optional[In]], Tuple[torch.Tensor, Out]] = (
            custom_model_fwd if custom_model_fwd else model
        )

        # 兼容旧版本的废弃字段
        self._batch_i: Optional[In] = None
        self._batch_ip1: Optional[In] = None
        self._batch_ip2: Optional[In] = None
        self._context: TrainPipelineContext = context_type(version=0)

    def detach(self) -> torch.nn.Module:
        """
        将模型从稀疏数据分发流水线中分离。
        Detaches the model from sparse data dist (SDD) pipeline.
        
        用户在训练后可能想要获取原始模型。原始model.forward之前被训练流水线修改过。
        A user might want to get the original model back after training. The original model.forward was previously
        modified by the train pipeline.
        
        参考/Reference: https://github.com/pytorch/torchrec/pull/2076
        
        返回/Returns:
            torch.nn.Module: 原始模型 / the original model
        """
        if self._pipelined_modules:
            _pipeline_detach_model(
                model=self._model,
                pipelined_modules=self._pipelined_modules,
                original_forwards=self._original_forwards,
                original_kjt_dist_forwards=self._original_kjt_dist_forwards,
                pipelined_postprocs=self._pipelined_postprocs,
            )

        self._model_attached = False
        return self._model

    def attach(
        self, model: Optional[torch.nn.Module] = None, sparse_dist: bool = True
    ) -> None:
        """
        将模型重新附加到流水线。
        Attach the model back to the pipeline.
        
        应与detach函数配合使用，当用户需要切换训练流水线时使用。
        Should be used with detach function when user wants to switch the train pipeline.
        
        参考/Reference: https://github.com/pytorch/torchrec/pull/2076
        """
        if model:
            self._model = model

        self._model_attached = True
        if self.contexts:
            self._pipeline_model(
                batch=self.batches[0] if sparse_dist else None,
                context=self.contexts[0],
                pipelined_forward=self._pipelined_forward_type,
            )
        else:
            # 在训练结束后附加模型，contexts为空
            # 重置_pipelined_modules，使_fill_pipeline在progress()中重新改写模型
            self._pipelined_modules = []
            self._pipelined_postprocs = []

    def _set_module_context(self, context: TrainPipelineContext) -> None:
        """
        为流水线模块设置上下文。
        Set context for pipelined modules.
        
        流水线模块是TorchRec的稀疏模块（如shardedEBC、shardedEC等）。
        Pipelined modules are TorchRec's sparse modules like shardedEBC, shardedEC, etc.
        
        forward函数在_rewrite_model调用中被替换为PipelinedForward。
        The forward function is swapped with a PipelinedForward in the _rewrite_model call.
        
        PipelinedForward需要上下文来正确执行前向行为。
        The PipelinedForward needs a context to correctly perform the forward behavior.
        """
        for module in self._pipelined_modules:
            module.forward.set_context(context)

        for postproc_module in self._pipelined_postprocs:
            # 确保下一次迭代的前向使用缓存结果
            postproc_module.set_context(context)

    def enqueue_batch(self, dataloader_iter: Iterator[In]) -> bool:
        """
        从数据加载器加载一个批次，复制到GPU，并创建上下文。
        Load a data batch from dataloader, and copy it from cpu to gpu.
        Also create the context for this batch.
        
        参数/Args:
            dataloader_iter: 数据加载器迭代器 / dataloader iterator
            
        返回/Returns:
            bool: 是否成功加载批次 / whether batch was loaded successfully
        """
        batch, context = self.copy_batch_to_gpu(dataloader_iter)
        if batch is None:
            return False
        self.batches.append(batch)
        self.contexts.append(context)

        return True

    def dequeue_batch(self) -> None:
        """
        从批次队列中移除已处理的批次。
        Remove a processed batch from the batch queue.
        
        同时更新模块上下文以匹配下一次前向传播。
        Also set the module context if applicable to match next forward pass.
        """
        self.batches.popleft()
        self.contexts.popleft()

        # 更新PipelinedForward上下文以匹配下一次前向传播
        if len(self.batches) >= 1:
            self._set_module_context(self.contexts[0])

    def fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        """
        填充流水线。在self.progress中调用。
        This function is called in self.progress (one of the main APIs for running train pipeline).
        
        Here we assume the max pipelined len(batches) == 2 (capacity), which will be the most common
        scenario during the full training job, when this function is effectively doing nothing.
        
        假设最大流水线的len(batches) == 2（容量），这是最常见的场景。
        There would only be two other scenarios:
        
        其他场景：
        - len(batches) == 0: 初始化流水线，填充两个批次，为第一个批次启动input_dist
            initialize the pipeline, fill in two batches, start input_dist for the first batch.
        - len(batches) == 1: 数据加载器停止，最后一个批次，不执行任何操作
            dataloader_iter stops, the last batch, do nothing
        """
        # 流水线已满（容量为2）
        if len(self.batches) >= 2:
            return

        # 执行流水线中的最后一个批次（只有一个批次时）
        if self.batches and self._execute_all_batches:
            return

        # 批次i：数据和上下文
        if not self.enqueue_batch(dataloader_iter):
            return

        # 修改（分片）稀疏模块的forward，并调用input_dist的第一部分
        self._init_pipelined_modules(
            self.batches[0],
            self.contexts[0],
            self._pipelined_forward_type,
        )
        # 执行input_dist的第二部分（第一部分在_init_pipelined_modules中调用）
        self.wait_sparse_data_dist(self.contexts[0])

        # 批次i+1
        if not self.enqueue_batch(dataloader_iter):
            return

    def _wait_for_batch(self) -> None:
        """
        等待批次在设备上可用。
        使用record_function进行性能分析标记。
        """
        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cast(In, self.batches[0]), self._data_dist_stream)

    def _backward(self, losses: torch.Tensor) -> None:
        """
        执行反向传播。
        Execute backward pass.
        
        参数/Args:
            losses: 损失张量，形状为(batch_size,) / loss tensor of shape (batch_size,)
        """
        with record_function("## backward ##"):
            torch.sum(losses, dim=0).backward()

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        """
        流水线的主要API，推进训练一步。
        Main API for advancing the pipeline by one training step.
        
        For TrainPipelineSparseDist, we assume the max pipelined batches == 3 (capacity):
        - batches[0]: current batch, for emb_lookup, output_dist, and fwd/bwd/opt (expecting input_dist)
        - batches[1]: next batch, for input_dist (expecting copied to device)
        - batches[2]: i+2 batch, for copy_batch_to_gpu (expecting non-exhausted dataloader iter)
        
        对于TrainPipelineSparseDist，假设最大流水线批次 == 3：
        - batches[0]: 当前批次，用于嵌入查找、output_dist和前向/反向/优化
        - batches[1]: 下一批次，用于input_dist
        - batches[2]: i+2批次，用于copy_batch_to_gpu
        
        参数/Args:
            dataloader_iter: 数据加载器迭代器 / dataloader iterator
            
        返回/Returns:
            Out: 当前批次的输出 / current batch output
        """
        # 确保模型已附加（防止用户忘记调用）
        if not self._model_attached:
            self.attach(self._model)

        # 仅在流水线开始时需要填充
        self.fill_pipeline(dataloader_iter)

        # 批次耗尽时停止
        if not self.batches:
            raise StopIteration

        # 设置模块上下文（向后兼容）
        self._set_module_context(self.contexts[0])

        # 梯度清零
        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        # 等待batches[0]在设备上可用
        self._wait_for_batch()

        # 启动batches[1]的稀疏数据分发（第一阶段）
        if len(self.batches) >= 2:
            self.start_sparse_data_dist(self.batches[1], self.contexts[1])

        # 批次i+2：加载数据并复制到GPU
        self.enqueue_batch(dataloader_iter)

        # 前向传播
        with record_function("## forward ##"):
            losses, output = self._model_fwd(self.batches[0])

        # 等待batches[1]的数据分发（第二阶段）
        if len(self.batches) >= 2:
            self.wait_sparse_data_dist(self.contexts[1])

        # 反向传播和优化（仅训练模式）
        if self._model.training:
            self._backward(losses)
            with record_function("## optimizer ##"):
                self._optimizer.step()

        # 出队当前批次
        self.dequeue_batch()
        return output

    def _create_context(self) -> TrainPipelineContext:
        """
        创建新的流水线上下文。
        Create a new pipeline context.
        
        返回/Returns:
            TrainPipelineContext: 新上下文，带有递增的索引 / new context with incrementing index
        """
        context = self._context_type(index=self._next_index, version=1)
        self._next_index += 1
        return context

    def _pipeline_model(
        self,
        batch: Optional[In],
        context: TrainPipelineContext,
        pipelined_forward: Type[PipelinedForward] = PipelinedForward,
    ) -> None:
        """
        改写模型以支持流水线。
        Rewrite the model to support pipelining.
        
        调用_rewrite_model函数修改模型forward。
        Calls _rewrite_model to modify model forward.
        """
        (
            self._pipelined_modules,
            self._model,
            self._original_forwards,
            self._pipelined_postprocs,
            _,
        ) = _rewrite_model(
            model=self._model,
            context=context,
            dist_stream=self._data_dist_stream,
            default_stream=torch.get_device_module(self._device).current_stream(),
            batch=batch,
            apply_jit=self._apply_jit,
            pipelined_forward=pipelined_forward,
            pipeline_postproc=self._pipeline_postproc,
        )
        # 初始化input_dist以覆盖input_dist forwards
        self.start_sparse_data_dist(batch, context)
        self._original_kjt_dist_forwards = _override_input_dist_forwards(
            self._pipelined_modules
        )

    def _init_pipelined_modules(
        self,
        batch: In,
        context: TrainPipelineContext,
        pipelined_forward: Type[PipelinedForward] = PipelinedForward,
    ) -> None:
        """
        检索流水线模块，初始化模块的input dists，并覆盖input dist forwards
        以支持融合splits集合操作。
        
        Retrieves the pipelined modules after overriding their forwards, initializes the
        modules' input dists, and overrides the input dist forwards to support fusing
        the splits collective in the input dist.
        """
        if self._pipelined_modules:
            # 模块已初始化，只需设置上下文并启动数据分发
            self._set_module_context(context)
            self.start_sparse_data_dist(batch, context)
            return

        # 首次调用，需要改写模型
        self._pipeline_model(batch, context, pipelined_forward)

    def copy_batch_to_gpu(
        self,
        dataloader_iter: Iterator[In],
    ) -> Tuple[Optional[In], Optional[TrainPipelineContext]]:
        """
        从数据加载器获取批次并移动到设备。
        Retrieves batch from dataloader and moves it to the provided device.
        
        参数/Args:
            dataloader_iter: 数据加载器迭代器 / dataloader iterator
            
        返回/Returns:
            Tuple[批次, 上下文]: 批次可能为None（数据加载器耗尽）/ batch may be None if dataloader exhausted
            
        异常/Raises:
            StopIteration: 如果数据加载器耗尽且execute_all_batches为False
            StopIteration: if the dataloader iterator is exhausted; unless execute_all_batches=True
        """
        context = self._create_context()
        with nvtx.annotate(f"## copy_batch_to_gpu {self._next_index} ##"):
            with self._stream_context(self._memcpy_stream):
                batch = self._next_batch(dataloader_iter)
                if batch is not None:
                    batch = _to_device(batch, self._device, non_blocking=True)
                elif not self._execute_all_batches:
                    raise StopIteration
                return batch, context

    def _next_batch(self, dataloader_iter: Iterator[In]) -> Optional[In]:
        """
        从数据加载器获取下一批次。
        Retrieves next batch from dataloader.
        
        防止在已耗尽的数据加载器上调用next，这可能导致挂起。
        Prevents calling `next` on an already exhausted dataloader, which can cause hanging.
        """
        if dataloader_iter is not self._dataloader_iter:
            self._dataloader_iter = dataloader_iter
            self._dataloader_exhausted = False

        if self._dataloader_exhausted:
            batch = None
        else:
            with record_function("## next_batch ##"):
                batch = next(dataloader_iter, None)
            if batch is None:
                self._dataloader_exhausted = True
        return batch

    def start_sparse_data_dist(
        self, batch: Optional[In], context: TrainPipelineContext
    ) -> None:
        """
        等待批次完成复制到GPU，然后启动input dist。
        Waits for batch to finish getting copied to GPU, then starts the input dist.
        """
        if batch is None:
            return
        with record_function(f"## start_sparse_data_dist {context.index} ##"):
            with self._stream_context(self._data_dist_stream):
                _wait_for_batch(batch, self._memcpy_stream)

                # 保存原始上下文
                original_contexts = [p.get_context() for p in self._pipelined_postprocs]

                # 临时设置为下一迭代的上下文以填充缓存
                for postproc_mod in self._pipelined_postprocs:
                    postproc_mod.set_context(context)
                _start_data_dist(self._pipelined_modules, batch, context)

                # 恢复模型前向的上下文
                for module, context in zip(
                    self._pipelined_postprocs, original_contexts
                ):
                    module.set_context(context)

    def wait_sparse_data_dist(self, context: TrainPipelineContext) -> None:
        """
        等待input dist splits请求获取input dist张量请求，并将其填充到上下文中。
        Waits on the input dist splits requests to get the input dist tensors requests,
        and populates the context with them.
        """
        with record_function(f"## wait_sparse_data_dist {context.index} ##"):
            with self._stream_context(self._data_dist_stream):
                for names, awaitable in context.fused_splits_awaitables:
                    for name, request in zip(names, awaitable.wait()):
                        context.input_dist_tensors_requests[name] = request
        context.input_dist_splits_requests.clear()
        context.fused_splits_awaitables.clear()

    # ========== 以下方法为向后兼容的废弃方法 ==========
    def _copy_batch_to_gpu(self, dataloader_iter: Iterator[In]) -> Optional[In]:
        """废弃/DEPRECATED: exists for backward compatibility on TrainPipelineContext.version 0"""
        self._set_module_context(self._context)
        batch, _ = self.copy_batch_to_gpu(dataloader_iter)
        return batch

    def _start_sparse_data_dist(self, batch: Optional[In]) -> None:
        """废弃/DEPRECATED: exists for backward compatibility"""
        self._set_module_context(self._context)
        self.start_sparse_data_dist(batch, self._context)

    def _wait_sparse_data_dist(self) -> None:
        """废弃/DEPRECATED: exists for backward compatibility"""
        self._set_module_context(self._context)
        with record_function("## wait_sparse_data_dist ##"):
            with self._stream_context(self._data_dist_stream):
                self._context.module_contexts = (
                    self._context.module_contexts_next_batch.copy()
                )
                self._context.input_dist_tensors_requests.clear()
                for names, awaitable in self._context.fused_splits_awaitables:
                    for name, request in zip(names, awaitable.wait()):
                        self._context.input_dist_tensors_requests[name] = request

    def _fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        """废弃/DEPRECATED: exists for backward compatibility"""
        if self._batch_i and self._batch_ip1:
            return
        if self._batch_i and self._execute_all_batches:
            return

        # 批次1
        self._batch_i = self._copy_batch_to_gpu(dataloader_iter)
        if self._batch_i is None:
            raise StopIteration

        self._init_pipelined_modules(self._batch_i, self._context)
        self._start_sparse_data_dist(self._batch_i)
        self._wait_sparse_data_dist()

        # 批次2
        self._batch_ip1 = self._copy_batch_to_gpu(dataloader_iter)


# ============================================
# 章节4: Prefetch训练流水线
# PrefetchTrainPipelineSparseDist - 支持预取优化的流水线
# ============================================

class PrefetchTrainPipelineSparseDist(TrainPipelineSparseDist[In, Out]):
    """
    带Prefetch优化的训练流水线。
    This pipeline overlaps device transfer, `ShardedModule.input_dist()`, and cache
    prefetching with forward and backward. This helps hide the all2all latency while
    preserving the training forward / backward ordering.
    
    在标准流水线的三阶段基础上，增加第四阶段：
    - Stage 4: 前向/反向计算 - 使用默认CUDA流 / forward, backward - uses default CUDA stream
    - Stage 3: Prefetch - 使用prefetch CUDA流 / prefetch - uses prefetch CUDA stream
    - Stage 2: ShardedModule.input_dist() - 使用data_dist CUDA流
    - Stage 1: 设备传输 - 使用memcpy CUDA流 / device transfer - uses memcpy CUDA stream
    
    Prefetch阶段用于从UVM/主机内存预取嵌入到GPU缓存，减少嵌入查找延迟。
    Prefetch stage is used to prefetch embeddings from UVM/host memory to GPU cache.
    
    参数/Args: 同TrainPipelineSparseDist，增加prefetch相关参数 / same as parent with prefetch params
    """
    
    # 使用PrefetchPipelinedForward替代标准PipelinedForward
    _pipelined_forward_type = PrefetchPipelinedForward

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        pipeline_postproc: bool = True,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
        prefetch_overlap_mode: Optional[str] = None,
        prefetch_debug: Optional[bool] = None,
        prefetch_debug_interval: Optional[int] = None,
    ) -> None:
        """
        初始化Prefetch训练流水线。
        Initialize Prefetch training pipeline.
        """
        # 调用父类初始化
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            execute_all_batches=execute_all_batches,
            apply_jit=apply_jit,
            context_type=PrefetchTrainPipelineContext,  # 使用Prefetch专用上下文
            pipeline_postproc=pipeline_postproc,
            custom_model_fwd=custom_model_fwd,
        )
        
        # 使用Prefetch专用上下文（废弃字段，向后兼容）
        self._context = PrefetchTrainPipelineContext(version=0)
        
        # 创建prefetch流
        self._prefetch_stream: Optional[torch.Stream] = (
            (torch.get_device_module(device).Stream())
            if self._device.type in ["cuda", "mtia", "npu"]
            else None
        )
        
        # 创建默认流（用于prefetch同步）
        self._default_stream: Optional[torch.Stream] = (
            (torch.get_device_module(self._device).Stream())
            if self._device.type in ["cuda", "mtia", "npu"]
            else None
        )
        
        # 设置prefetch调试模式
        self._prefetch_debug: bool = (
            prefetch_debug
            if prefetch_debug is not None
            else os.getenv("HSTU_PREFETCH_DEBUG", "1").lower() in ("1", "true")
        )
        
        # 设置调试日志间隔
        self._prefetch_debug_interval: int = (
            prefetch_debug_interval
            if prefetch_debug_interval is not None
            else int(os.getenv("HSTU_PREFETCH_DEBUG_INTERVAL", "20"))
        )
        
        # 设置重叠模式
        self._prefetch_overlap_mode: str = (
            prefetch_overlap_mode.lower()
            if prefetch_overlap_mode is not None
            else os.getenv("HSTU_PREFETCH_OVERLAP_MODE", "safe").lower()
        )
        
        # 激进模式：允许更多重叠，但可能有风险
        self._prefetch_overlap_aggressive: bool = (
            self._prefetch_overlap_mode == "aggressive"
        )
        
        # 进度计数
        self._progress_step: int = 0
        self._batch_ip3: Optional[In] = None
        
        # 打印调试信息
        if self._prefetch_debug:
            logger.warning(
                "[prefetch-debug] overlap_mode=%s (aggressive=%s).",
                self._prefetch_overlap_mode,
                self._prefetch_overlap_aggressive,
            )

    def _fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        """
        填充Prefetch流水线。
        Fill the prefetch pipeline.
        
        与标准流水线不同，这里需要处理三个批次：batch_i, batch_ip1, batch_ip2
        Unlike standard pipeline, this handles three batches: batch_i, batch_ip1, batch_ip2
        """
        # 流水线已填充
        if self._batch_i and self._batch_ip1 and self._batch_ip2:
            return
        # 执行流水线中的最后一个批次
        if self._execute_all_batches and (self._batch_i or self._batch_ip1):
            return

        # 批次1
        self._batch_i = self._copy_batch_to_gpu(dataloader_iter)
        if self._batch_i is None:
            raise StopIteration

        self._init_pipelined_modules(
            self._batch_i,
            self._context,
            self._pipelined_forward_type,
        )
        self._start_sparse_data_dist(self._batch_i)
        self._wait_sparse_data_dist()
        self._prefetch(self._batch_i)  # 预取第一个批次

        # 批次2
        self._batch_ip1 = self._copy_batch_to_gpu(dataloader_iter)
        self._start_sparse_data_dist(self._batch_ip1)

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        """
        Prefetch流水线的progress实现。
        Prefetch pipeline progress implementation.
        
        流程/Flow:
        1. 填充流水线（3个批次）/ fill pipeline (3 batches)
        2. 梯度清零 / zero grad
        3. 等待批次就绪 / wait for batch
        4. 复制下一批次到GPU / copy next batch to GPU
        5. 等待数据分发 / wait for data dist
        6. 前向传播 / forward
        7. 预取下一批次 / prefetch next batch
        8. 反向传播 / backward
        9. 优化器步进 / optimizer step
        10. 启动下一批次的数据分发 / start data dist for next batch
        11. 更新批次引用 / update batch references
        """
        self._fill_pipeline(dataloader_iter)

        # 梯度清零
        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        # 等待批次就绪
        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cast(In, self._batch_i), self._prefetch_stream)

        # 复制下一批次
        self._batch_ip2 = self._copy_batch_to_gpu(dataloader_iter)

        # 等待数据分发完成
        self._wait_sparse_data_dist()
        
        # 前向传播
        with record_function("## forward ##"):
            losses, output = self._model_fwd(self._batch_i)

        # 预取下一批次
        self._prefetch(self._batch_ip1)

        # 反向传播和优化（仅训练模式）
        if self._model.training:
            with record_function("## backward ##"):
                torch.sum(losses, dim=0).backward()

            with record_function("## optimizer ##"):
                self._optimizer.step()

        # 启动下一批次的数据分发
        self._start_sparse_data_dist(self._batch_ip2)

        # 更新批次引用：i -> i+1, i+1 -> i+2
        self._batch_i = self._batch_ip1
        self._batch_ip1 = self._batch_ip2

        return output

    def _prefetch(self, batch: Optional[In]) -> None:
        """
        等待input dist完成，然后预取数据。
        Waits for input dist to finish, then prefetches data.
        
        参数/Args:
            batch: 要预取的批次 / batch to prefetch
        """
        if batch is None:
            return
        
        # 清空预取相关的上下文
        self._context.module_input_post_prefetch.clear()
        self._context.module_contexts_post_prefetch.clear()

        with record_function("## sharded_module_prefetch ##"):
            with self._stream_context(self._prefetch_stream):
                # 记录批次到当前流
                batch.record_stream(
                    torch.get_device_module(self._device).current_stream()
                )
                
                # 调用预取嵌入函数
                data_per_pipelined_module = _prefetch_embeddings(
                    batch,
                    self._context,
                    self._pipelined_modules,
                    self._device,
                    self._stream_context,
                    self._data_dist_stream,
                    self._default_stream,
                )
                
                # 将预取数据保存到上下文
                for sharded_module in self._pipelined_modules:
                    forward = sharded_module.forward
                    data = data_per_pipelined_module[forward._name]
                    self._context.module_input_post_prefetch[forward._name] = data
                    self._context.module_contexts_post_prefetch[
                        forward._name
                    ] = self._context.module_contexts.pop(forward._name)


# ============================================
# 章节5: Megatron变长序列训练流水线
# JaggedMegatronTrainPipelineSparseDist - 适配Megatron的变长序列版本
# ============================================

class JaggedMegatronTrainPipelineSparseDist(TrainPipelineSparseDist[In, Out]):
    """
    适配Megatron的变长序列训练流水线。
    Jagged sequence training pipeline adapted for Megatron.
    
    继承自TrainPipelineSparseDist，增加了：
    Inherits from TrainPipelineSparseDist, adds:
    - Megatron特定的梯度处理（finalize_model_grads）/ Megatron-specific gradient handling
    - 损失计算和分布式归约 / loss computation and distributed reduction
    - 变长序列支持（jagged tensors）/ jagged sequence support
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        pipeline_postproc: bool = False,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
    ) -> None:
        """
        初始化Megatron变长序列训练流水线。
        Initialize Megatron jagged sequence training pipeline.
        """
        super().__init__(
            model,
            optimizer,
            device,
            execute_all_batches,
            apply_jit,
            TrainPipelineContext,
            pipeline_postproc,
            custom_model_fwd,
        )

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        """
        Megatron变长序列版本的progress实现。
        Megatron jagged sequence version of progress.
        
        与父类的主要区别/Differences from parent:
        1. 使用nvtx进行更细粒度的性能分析 / uses nvtx for finer-grained profiling
        2. 支持Megatron的zero_grad_buffer / supports Megatron's zero_grad_buffer
        3. 处理变长序列的损失计算和分布式归约 / handles jagged loss computation and distributed reduction
        4. 调用finalize_model_grads进行梯度处理 / calls finalize_model_grads for gradient handling
        """
        # 确保模型已附加
        if not self._model_attached:
            self.attach(self._model)

        # 填充流水线
        self.fill_pipeline(dataloader_iter)

        # 批次耗尽时停止
        if not self.batches:
            raise StopIteration

        # 设置模块上下文
        self._set_module_context(self.contexts[0])

        # 梯度清零
        if self._model.training:
            with nvtx.annotate("## zero_grad ##"):
                # 如果模型支持zero_grad_buffer，调用它
                if hasattr(self._model.module, "zero_grad_buffer"):
                    self._model.module.zero_grad_buffer()
                self._optimizer.zero_grad()

        # 等待批次就绪
        with nvtx.annotate("## wait_for_batch ##"):
            self._wait_for_batch()

        # 启动下一批次的数据分发
        with nvtx.annotate("## start_sparse_data_dist ##"):
            if len(self.batches) >= 2:
                self.start_sparse_data_dist(self.batches[1], self.contexts[1])

        # 复制下一批次
        with nvtx.annotate("## enqueue_batch ##"):
            self.enqueue_batch(dataloader_iter)

        # 前向传播
        with nvtx.annotate("## forward ##"):
            losses, output = self._model_fwd(self.batches[0])
            
        # 损失后处理：检查NaN，计算本地损失和token数
        with nvtx.annotate("## loss postprocess ##"):
            collective_assert(not torch.isnan(losses).any(), "loss has nan value")
            local_tokens = torch.tensor(losses.size(0), device=self._device).float()
            local_loss = torch.cat([torch.sum(losses).view(1), local_tokens.view(1)])
            reporting_loss = local_loss.clone().detach()
            
            # 分布式归约损失（跨数据并行组）
            torch.distributed.all_reduce(
                reporting_loss, group=parallel_state.get_data_parallel_group()
            )
            
        # 等待数据分发完成
        if len(self.batches) >= 2:
            with nvtx.annotate("## wait_sparse_data_dist ##"):
                self.wait_sparse_data_dist(self.contexts[1])

        # 反向传播和优化
        if self._model.training:
            with nvtx.annotate("## backward ##"):
                dp_size = parallel_state.get_data_parallel_world_size()
                # 处理跨DP rank的不均匀变长大小
                local_loss_average = local_loss[0] / reporting_loss[1] * dp_size
                local_loss_average.backward()

            # 调用finalize_model_grads处理梯度（Megatron特定）
            with nvtx.annotate("## finalize_model_grads ##"):
                if isinstance(self._model.module, DistributedDataParallel):
                    finalize_model_grads([self._model.module], None)
                    
            # 优化器步进
            with nvtx.annotate("## optimizer ##"):
                self._optimizer.step()
                
        # 出队
        self.dequeue_batch()
        return reporting_loss, output


# ============================================
# 章节6: Megatron Prefetch训练流水线
# JaggedMegatronPrefetchTrainPipelineSparseDist - 结合Megatron和Prefetch优化
# ============================================

class JaggedMegatronPrefetchTrainPipelineSparseDist(
    PrefetchTrainPipelineSparseDist[In, Out]
):
    """
    结合Megatron特性和Prefetch优化的变长序列训练流水线。
    Jagged sequence training pipeline combining Megatron features and Prefetch optimization.
    
    这是最完整的流水线实现，支持：
    This is the most complete pipeline implementation, supporting:
    - 变长序列处理 / jagged sequence handling
    - Prefetch优化 / prefetch optimization
    - Megatron梯度处理 / Megatron gradient handling
    - 激进的/安全的重叠模式 / aggressive/safe overlap modes
    """
    
    def __init__(
        self,
        model: torch.nn.Module,  # might be wrapped by DistributedModelParallel / 可能被DMP包装
        optimizer: torch.optim.Optimizer,  # dense optimizer, might be megatron optimizer / 稠密优化器，可能是Megatron优化器
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        pipeline_postproc: bool = True,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
        prefetch_overlap_mode: Optional[str] = None,
        prefetch_debug: Optional[bool] = None,
        prefetch_debug_interval: Optional[int] = None,
    ) -> None:
        """
        初始化Megatron Prefetch训练流水线。
        Initialize Megatron Prefetch training pipeline.
        """
        super().__init__(
            model,
            optimizer,
            device,
            execute_all_batches,
            apply_jit,
            pipeline_postproc,
            custom_model_fwd,
            prefetch_overlap_mode,
            prefetch_debug,
            prefetch_debug_interval,
        )

    def progress(self, dataloader_iter: Iterator[In]) -> Tuple[torch.Tensor, Out]:
        """
        Megatron Prefetch流水线的progress实现。
        Megatron Prefetch pipeline progress implementation.
        
        这是完整的训练步骤，包含：
        This is the complete training step, containing:
        1. 流水线填充 / pipeline fill
        2. 梯度清零 / zero grad
        3. 等待批次 / wait for batch
        4. 复制批次 / copy batch
        5. 等待数据分发 / wait for data dist
        6. 前向传播 / forward
        7. 损失后处理 / loss postprocess
        8. Prefetch / prefetch
        9. （可选）激进的input_dist重叠 / (optional) aggressive input_dist overlap
        10. 反向传播前的sync（处理prefetch和反向的竞态条件）/ pre-backward sync (handle prefetch-backward race)
        11. 反向传播和梯度处理 / backward and gradient handling
        12. 优化器步进 / optimizer step
        13. （如果未激进重叠）启动input_dist / (if not aggressive) start input_dist
        14. 更新批次引用 / update batch references
        """
        # 填充流水线
        self._fill_pipeline(dataloader_iter)

        # 梯度清零
        if self._model.training:
            with nvtx.annotate("## zero_grad ##"):
                if hasattr(self._model.module, "zero_grad_buffer"):
                    self._model.module.zero_grad_buffer()
                self._optimizer.zero_grad()

        # 等待批次就绪（在prefetch流上）
        with nvtx.annotate("## wait_for_batch ##"):
            _wait_for_batch(cast(In, self._batch_i), self._prefetch_stream)

        # 复制下一批次到GPU
        with nvtx.annotate("## copy_batch_to_gpu ##"):
            self._batch_ip2 = self._copy_batch_to_gpu(dataloader_iter)
            
        # 等待数据分发完成
        with nvtx.annotate("## wait_sparse_data_dist ##"):
            self._wait_sparse_data_dist()
            
        # 前向传播
        reporting_loss = None
        with nvtx.annotate("## forward ##"):
            losses, output = self._model_fwd(self._batch_i)
            
        # 损失后处理
        with nvtx.annotate("## loss postprocess ##"):
            collective_assert(not torch.isnan(losses).any(), "loss has nan value")
            local_tokens = torch.tensor(losses.size(0), device=self._device).float()
            local_loss = torch.cat([torch.sum(losses).view(1), local_tokens.view(1)])
            reporting_loss = local_loss.clone().detach()

            # 分布式归约
            torch.distributed.all_reduce(
                reporting_loss, group=parallel_state.get_data_parallel_group()
            )

        # 预取下一批次
        with nvtx.annotate("## prefetch ##"):
            self._prefetch(self._batch_ip1)

        # 激进的重叠模式：提前启动input_dist
        started_input_dist = False
        if self._model.training and self._prefetch_overlap_aggressive:
            with nvtx.annotate("## input_dist ##"):
                self._start_sparse_data_dist(self._batch_ip2)
                started_input_dist = True

        # 反向传播（仅训练模式）
        if self._model.training:
            # prefetch => 加载到缓存 & 可能使缓存失效 & 刷新到主机
            # 反向 => 读写缓存/主机
            # 为避免竞态条件，强制sync：prefetch应完成
            sync_begin = time.perf_counter()
            
            if self._prefetch_overlap_aggressive:
                # 激进模式：跳过sync（假设风险可控）
                if self._prefetch_debug and (
                    (self._progress_step + 1) % self._prefetch_debug_interval == 0
                ):
                    logger.warning(
                        "[prefetch-debug] overlap_mode=aggressive 跳过反向prefetch同步 step=%d.",
                        self._progress_step + 1,
                    )
            elif self._prefetch_stream is not None:
                # 安全模式：等待prefetch流完成
                torch.get_device_module(self._device).current_stream().wait_stream(
                    self._prefetch_stream
                )
            elif self._prefetch_debug:
                logger.warning(
                    "[prefetch-debug] prefetch_stream在设备%s上为None；跳过wait_stream同步.",
                    self._device,
                )
                
            sync_ms = (time.perf_counter() - sync_begin) * 1000.0
            self._progress_step += 1
            
            # 打印调试信息
            if self._prefetch_debug and (
                self._progress_step % self._prefetch_debug_interval == 0
            ):
                logger.warning(
                    "[prefetch-debug] step=%d sync_wait_ms=%.3f has_prefetch_stream=%s has_data_dist_stream=%s",
                    self._progress_step,
                    sync_ms,
                    self._prefetch_stream is not None,
                    self._data_dist_stream is not None,
                )
                
            # 反向传播
            with nvtx.annotate("## backward ##"):
                dp_size = parallel_state.get_data_parallel_world_size()
                # 处理跨DP rank的不均匀变长大小
                # 注意：不能完全解决，需要sum reduce才能彻底解决
                # 参考：https://github.com/NVIDIA/Megatron-LM/blob/v0.12.0rc3/megatron/core/distributed/distributed_data_parallel.py#L237-L240
                local_loss_average = local_loss[0] / reporting_loss[1] * dp_size
                local_loss_average.backward()

                # self._model是DistributedModelParallel
                # self._model.module可能是DistributedDataParallel
                if isinstance(self._model.module, DistributedDataParallel):
                    finalize_model_grads([self._model.module], None)

            # 优化器步进
            with nvtx.annotate("## optimizer ##"):
                self._optimizer.step()

        # 如果未在激进模式下启动，现在启动input_dist
        if not started_input_dist:
            with nvtx.annotate("## input_dist ##"):
                self._start_sparse_data_dist(self._batch_ip2)

        # 更新批次引用
        self._batch_i = self._batch_ip1
        self._batch_ip1 = self._batch_ip2

        return reporting_loss, output


# ============================================
# 章节7: 无流水线训练
# JaggedMegatronTrainNonePipeline - 不使用流水线的简单训练
# ============================================

class JaggedMegatronTrainNonePipeline:
    """
    无流水线训练类。
    Train pipeline without any pipelining.
    
    这是最简化的训练实现，不使用任何流水线重叠。
    This is the simplest training implementation without any pipeline overlapping.
    
    适用于：
    Suitable for:
    - 调试 / debugging
    - 不支持流水线的场景 / scenarios where pipelining is not supported
    - 对比实验 / comparison experiments
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        """
        初始化无流水线训练器。
        Initialize non-pipeline trainer.
        
        参数/Args:
            model: 模型 / model
            optimizer: 优化器 / optimizer
            device: 设备 / device
        """
        self._model = model
        self._optimizer = optimizer
        self._device = device

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        """
        简单的训练步骤，按顺序执行：
        Simple training step executed sequentially:
        1. 梯度清零 / zero grad
        2. H2D数据传输 / H2D data transfer
        3. 前向传播 / forward
        4. 损失计算和分布式归约 / loss computation and distributed reduction
        5. 反向传播 / backward
        6. 梯度处理 / gradient handling
        7. 优化器步进 / optimizer step
        """
        # 获取数据并行大小（用于损失平均）
        dp_size = parallel_state.get_data_parallel_world_size() * 1.0
        
        # 梯度清零
        with nvtx.annotate("## zero_grad ##"):
            if hasattr(self._model.module, "zero_grad_buffer"):
                self._model.module.zero_grad_buffer()
            self._optimizer.zero_grad()
            
        # 主机到设备数据传输
        with nvtx.annotate("## H2D ##"):
            batch = next(dataloader_iter).to(self._device)

        # 前向传播
        with nvtx.annotate("## forward ##"):
            losses, output = self._model(batch)

        # 损失后处理
        with nvtx.annotate("## loss postprocess ##"):
            collective_assert(not torch.isnan(losses).any(), "loss has nan value")
            local_tokens = torch.tensor(
                losses.size(0), device=self._device
            ).float()
            local_loss = torch.cat([torch.sum(losses).view(1), local_tokens.view(1)])
            reporting_loss = local_loss.clone().detach()
            
            # 分布式归约：[allreduced_sum_loss, allreduced_sum_tokens]
            torch.distributed.all_reduce(
                reporting_loss, group=parallel_state.get_data_parallel_group()
            )
            
        # 反向传播（仅训练模式）
        if self._model.training:
            with nvtx.annotate("## backward ##"):
                local_loss_average = local_loss[0] / reporting_loss[1] * dp_size
                local_loss_average.backward()
            
            # 当reshard_after_forward为True时unshard
            from torch.distributed.fsdp import FSDPModule
            fsdp_root = self._model.module
            for module in fsdp_root.modules():
                if isinstance(module, FSDPModule):
                    module.unshard()

            # 梯度处理
            with nvtx.annotate("## finalize_model_grads ##"):
                if isinstance(self._model.module, DistributedDataParallel):
                    finalize_model_grads([self._model.module], None)

            # 优化器步进
            with nvtx.annotate("## optimizer step ##"):
                self._optimizer.step()

        return reporting_loss, output
