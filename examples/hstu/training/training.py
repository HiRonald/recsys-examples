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

# ============================================
# 章节1: 导入模块
# 本文件实现训练循环、评估和检查点管理
# ============================================

import os
from itertools import chain, count, cycle, islice
from typing import Iterator, Optional, Union

# 导入检查点和工具模块
import commons.checkpoint as checkpoint
import torch
import torch_npu
import torch.distributed as dist

from commons.checkpoint import get_unwrapped_module
from commons.utils.gpu_timer import GPUTimer
from commons.utils.logger import print_rank_0
from commons.utils.stringify import stringify_dict
from megatron.core import parallel_state

# 导入模型和指标模块
from model import RankingGR, RetrievalGR
from modules.metrics import RetrievalTaskMetricWithSampling

# 导入训练流水线类
from pipeline.train_pipeline import (
    JaggedMegatronPrefetchTrainPipelineSparseDist,
    JaggedMegatronTrainNonePipeline,
    JaggedMegatronTrainPipelineSparseDist,
)

# 导入参数类和工具
from training.gin_config_args import TrainerArgs
from training.utils import cal_flops


# ============================================
# 章节2: 模型评估函数
# ============================================

def evaluate(
    pipeline: Union[
        JaggedMegatronPrefetchTrainPipelineSparseDist,
        JaggedMegatronTrainNonePipeline,
        JaggedMegatronTrainPipelineSparseDist,
    ],
    stateful_metric_module: torch.nn.Module,
    trainer_args: TrainerArgs,
    eval_loader: torch.utils.data.DataLoader,
):
    """
    评估模型在验证集上的性能。
    Evaluate model performance on validation set.
    
    流程/Flow:
    1. 遍历评估数据加载器 / iterate eval dataloader
    2. 使用pipeline.progress进行前向传播 / forward with pipeline.progress
    3. 计算评估指标 / compute evaluation metrics
    4. 打印评估结果 / print eval results
    
    参数/Args:
        pipeline: 训练流水线 / training pipeline
        stateful_metric_module: 有状态指标模块 / stateful metric module
        trainer_args: 训练参数 / training args
        eval_loader: 评估数据加载器 / eval dataloader
    """
    eval_iter = 0
    
    # 确定评估迭代次数（可能限制max_eval_iters）
    max_eval_iters = trainer_args.max_eval_iters or len(eval_loader)
    max_eval_iters = min(max_eval_iters, len(eval_loader))
    
    # 创建评估加载器的副本，避免修改原始加载器
    iterated_eval_loader = islice(eval_loader, len(eval_loader))
    
    # 禁用梯度计算进行评估
    with torch.no_grad():
        for i in range(max_eval_iters):
            eval_iter += 1
            
            # 使用pipeline进行前向传播（复用训练逻辑）
            reporting_loss, (_, logits, labels, _) = pipeline.progress(
                iterated_eval_loader
            )
            
            # 更新指标模块
            stateful_metric_module(logits, labels)
            
        # 计算最终指标（compute会重置状态）
        if isinstance(stateful_metric_module, RetrievalTaskMetricWithSampling):
            # 检索任务的特殊处理：需要导出嵌入
            retrieval_gr = get_unwrapped_module(pipeline._model)
            export_table_name = retrieval_gr.get_item_feature_table_name()
            eval_metric_dict, _, _ = stateful_metric_module.compute(
                *retrieval_gr._embedding_collection.export_local_embedding(
                    export_table_name
                ),
            )
        else:
            # 排序任务：直接计算指标
            eval_metric_dict = stateful_metric_module.compute()
            
        # 获取数据并行大小
        dp_size = parallel_state.get_data_parallel_world_size()
        
    # 打印评估结果
    # TODO: 修复不完整批次时的样本数计算
    print_rank_0(
        f"[eval] [eval {eval_iter * dp_size * trainer_args.eval_batch_size} users]:\n    "
        + stringify_dict(eval_metric_dict, prefix="Metrics", sep="\n    ")
    )


# ============================================
# 章节3: 检查点加载和保存
# ============================================

def maybe_load_ckpts(
    ckpt_load_dir: str,
    model: Union[RankingGR, RetrievalGR],
    dense_optimizer: Optional[torch.optim.Optimizer] = None,
):
    """
    如果指定了目录，则加载检查点。
    Load checkpoint if directory is specified.
    
    参数/Args:
        ckpt_load_dir: 检查点加载目录 / checkpoint load dir, empty means no load
        model: 模型 / model
        dense_optimizer: 稠密优化器（可选）/ dense optimizer (optional)
    """
    if ckpt_load_dir == "":
        return

    # 验证目录存在
    assert os.path.exists(
        ckpt_load_dir
    ), f"ckpt_load_dir {ckpt_load_dir} does not exist"

    print_rank_0(f"Loading checkpoints from {ckpt_load_dir}")
    checkpoint.load(ckpt_load_dir, model, dense_optimizer=dense_optimizer)
    print_rank_0(f"Checkpoints loaded!!")


def save_ckpts(
    ckpt_save_dir: str,
    model: Union[RankingGR, RetrievalGR],
    dense_optimizer: Optional[torch.optim.Optimizer] = None,
):
    """
    保存模型和优化器状态到检查点目录。
    Save model and optimizer state to checkpoint directory.
    
    参数/Args:
        ckpt_save_dir: 检查点保存目录 / checkpoint save dir
        model: 模型 / model
        dense_optimizer: 稠密优化器（可选）/ dense optimizer (optional)
    """
    print_rank_0(f"Saving checkpoints to {ckpt_save_dir}")
    import shutil

    # rank 0负责创建目录
    if dist.get_rank() == 0:
        if os.path.exists(ckpt_save_dir):
            shutil.rmtree(ckpt_save_dir)
        try:
            os.makedirs(ckpt_save_dir, exist_ok=True)
        except Exception as e:
            raise Exception("can't build path:", ckpt_save_dir) from e
            
    # 等待所有rank，确保目录创建完成
    dist.barrier(device_ids=[torch_npu.npu.current_device()])
    
    # 保存检查点
    checkpoint.save(ckpt_save_dir, model, dense_optimizer=dense_optimizer)
    print_rank_0(f"Checkpoints saved!!")


# ============================================
# 章节4: 批次分片工具函数
# ============================================

# TODO: Python 3.12+可使用itertools.batched
# 当前实现为兼容旧版本Python
def batched(it: Iterator, n: int):
    """
    将迭代器分片为大小为n的批次。
    Split iterator into batches of size n.
    
    参数/Args:
        it: 输入迭代器 / input iterator
        n: 批次大小 / batch size
        
    返回/Returns:
        产生大小为n的批次迭代器 / iterator yielding batches of size n
    """
    assert n >= 1
    for x in it:
        yield chain((x,), islice(it, n - 1))


# ============================================
# 章节5: 主训练函数
# ============================================

def train_with_pipeline(
    pipeline: Union[
        JaggedMegatronPrefetchTrainPipelineSparseDist,
        JaggedMegatronTrainNonePipeline,
        JaggedMegatronTrainPipelineSparseDist,
    ],
    stateful_metric_module: torch.nn.Module,
    trainer_args: TrainerArgs,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    dense_optimizer: torch.optim.Optimizer,
):
    """
    使用训练流水线执行完整训练。
    Execute full training with the pipeline.
    
    这是训练的主循环，包含：
    This is the main training loop, containing:
    1. 训练迭代 / training iterations
    2. 定期评估 / periodic evaluation
    3. 检查点保存 / checkpoint saving
    4. 日志记录和性能统计 / logging and performance stats
    5. NPU Profiler支持（可选）/ NPU Profiler support (optional)
    
    参数/Args:
        pipeline: 训练流水线实例 / pipeline instance
        stateful_metric_module: 有状态指标模块 / stateful metric module
        trainer_args: 训练参数 / training args
        train_loader: 训练数据加载器 / train dataloader
        eval_loader: 评估数据加载器 / eval dataloader
        dense_optimizer: 稠密优化器 / dense optimizer
    """
    # 创建GPU计时器
    gpu_timer = GPUTimer()
    
    # 确定最大训练迭代次数
    max_train_iters = trainer_args.max_train_iters or len(train_loader)
    
    # 启动计时器
    gpu_timer.start()
    last_td = 0  # 上次记录的时间差
    
    # 用于计算FLOPS的统计数据
    ddp_seqlens = []
    ddp_num_contextuals = []
    ddp_num_candidates = []
    
    # 使用GPU张量记录token数（避免D2H拷贝）
    tokens_logged = torch.zeros(1, device=pipeline._device).float()
    
    # 限制训练迭代次数
    # 支持max_train_iters > n_batches（多轮训练）
    train_loader_iter = islice(cycle(iter(train_loader)), max_train_iters)

    # 计算评估间隔
    n = trainer_args.eval_interval if trainer_args.eval_interval else max_train_iters
    
    # 将数据加载器分割为多个切片，每个切片包含n个批次
    iter_slices = batched(train_loader_iter, n)
    start_iter = 0
    
    # 设置模型为训练模式
    pipeline._model.train()

    # ============================================
    # NPU Profiler配置（可选）
    # 通过环境变量NPU_PROFILE控制
    # ============================================
    PROFILE_ENABLE = os.environ.get("NPU_PROFILE", "0").lower() in ("1", "true")
    if PROFILE_ENABLE:
        # 配置实验性选项
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=[
                torch_npu.profiler.ExportType.Text,
                torch_npu.profiler.ExportType.Db,
            ],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=False,
            data_simplification=False,
            record_op_args=False,
            gc_detect_threshold=None,
        )

        # 创建Profiler
        prof = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
            ],
            schedule=torch_npu.profiler.schedule(
                wait=10,      # 前10步不分析
                warmup=1,     # 第11步热身
                active=3,     # 分析3步
                repeat=1,     # 重复1次
                skip_first=1  # 跳过第1步
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
            with_modules=True,
            with_flops=False,
            experimental_config=experimental_config
        )

        prof.start()  # 启动Profiler

    # ============================================
    # 主训练循环
    # ============================================
    for batched_iterator in iter_slices:
        # 遍历每个切片（评估间隔内的批次）
        for train_iter in count(start_iter):
            
            # 性能分析起始点
            if trainer_args.profile and train_iter == trainer_args.profile_step_start:
                dist.barrier(device_ids=[torch_npu.npu.current_device()])
                
            # 性能分析结束点
            if trainer_args.profile and train_iter == trainer_args.profile_step_end:
                dist.barrier(device_ids=[torch_npu.npu.current_device()])
                
            # 定期保存检查点
            if (
                train_iter * trainer_args.ckpt_save_interval > 0
                and train_iter % trainer_args.ckpt_save_interval == 0
            ):
                save_path = os.path.join(
                    trainer_args.ckpt_save_dir, f"iter{train_iter}"
                )
                save_ckpts(save_path, pipeline._model, dense_optimizer)
                
            try:
                # 执行一个训练步骤
                reporting_loss, (
                    local_loss,
                    logits,
                    labels,
                    (ddp_seqlen, ddp_num_contextual, ddp_num_candidate),
                ) = pipeline.progress(batched_iterator)
                
                # 收集统计数据用于FLOPS计算
                ddp_seqlens.append(ddp_seqlen.view(-1))
                ddp_num_contextuals.append(ddp_num_contextual.view(-1))
                ddp_num_candidates.append(ddp_num_candidate.view(-1))
                
                # 累加token数
                tokens_logged += reporting_loss[1]
                
                # Profiler步进
                if PROFILE_ENABLE:
                    prof.step()
                    
            except StopIteration:
                # 当前切片完成，记录起始迭代
                start_iter = train_iter
                break
                
            # ============================================
            # 日志记录
            # ============================================
            if train_iter > 0 and (train_iter + 1) % trainer_args.log_interval == 0:
                gpu_timer.stop()
                cur_td = gpu_timer.elapsed_time() - last_td
                
                # 计算FLOPS
                flops = cal_flops(
                    get_unwrapped_module(pipeline._model)._hstu_config,
                    seqlens=ddp_seqlens,
                    num_contextuals=ddp_num_contextuals,
                    num_candidates=ddp_num_candidates,
                )
                
                # 打印训练日志
                print_rank_0(
                    f"[train] [iter {train_iter}, tokens {int(tokens_logged.item())}, "
                    f"elapsed_time {cur_td:.2f} ms, achieved FLOPS {flops / cur_td / 1e9:.2f} TFLOPS]: "
                    f"loss {reporting_loss[0] / reporting_loss[1]:.6f}"
                )
                
                # 重置统计
                last_td = cur_td + last_td
                tokens_logged.zero_()
                ddp_seqlens = []
                ddp_num_contextuals = []
                ddp_num_candidates = []
                
        # ============================================
        # 定期评估
        # ============================================
        if train_iter > 0 and train_iter % trainer_args.eval_interval == 0:
            pipeline._model.eval()  # 切换到评估模式
            evaluate(
                pipeline,
                stateful_metric_module,
                trainer_args=trainer_args,
                eval_loader=eval_loader,
            )
            pipeline._model.train()  # 切回训练模式
            
    # 停止Profiler
    if PROFILE_ENABLE:
        prof.stop()
