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
# 本文件是排序任务（Ranking）预训练的主入口，实现了基于HSTU的分布式训练流程
# ============================================

import warnings
import sysconfig
import os

# 忽略所有FutureWarnings和SyntaxWarnings，避免训练过程中打印过多的警告信息
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

import argparse  # 命令行参数解析
from dataclasses import dataclass  # 数据类装饰器，用于简化配置类的定义
from functools import partial  # 偏函数，用于固定部分参数
from typing import List, Tuple, cast  # 类型注解工具

# 导入自定义初始化模块，用于分布式环境设置
import commons.utils.initialize as init

# gin-config用于配置注入，允许通过配置文件动态设置参数
import gin

import torch  # PyTorch深度学习框架
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
import torch_npu  # NPU（昇腾）设备支持

# MindSpeed适配器，用于在NPU上运行Megatron
import mindspeed.megatron_adaptor

# 导入日志工具，仅在rank 0打印日志
from commons.utils.logger import print_rank_0

# 导入配置类和分布式相关模块
from configs import RankingConfig
from distributed.sharding import make_optimizer_and_shard, make_optimizer_and_shard_fsdp2

# Megatron分布式并行状态管理
from megatron.core import parallel_state

# 导入模型和评估指标模块
from model import get_ranking_model
from modules.metrics import get_multi_event_metric_module

# 导入训练流水线相关类
# JaggedMegatronPrefetchTrainPipelineSparseDist: 带Prefetch优化的变长序列训练流水线
# JaggedMegatronTrainNonePipeline: 无流水线模式的训练
# JaggedMegatronTrainPipelineSparseDist: 标准稀疏分布式训练流水线
from pipeline.train_pipeline import (
    JaggedMegatronPrefetchTrainPipelineSparseDist,
    JaggedMegatronTrainNonePipeline,
    JaggedMegatronTrainPipelineSparseDist,
)

# 导入训练相关的参数类和工具函数
from training import (
    NetworkArgs,           # 网络架构参数
    DatasetArgs,           # 数据集参数
    EmbeddingArgs,         # 嵌入层参数
    OptimizerArgs,         # 优化器参数
    TensorModelParallelArgs,  # 张量模型并行参数
    TrainerArgs,           # 训练器参数
    create_dynamic_optitons_dict,  # 创建动态嵌入选项字典
    create_embedding_configs,        # 创建嵌入配置
    create_hstu_config,              # 创建HSTU模型配置
    create_optimizer_params,         # 创建优化器参数
    get_data_loader,                 # 获取数据加载器
    get_dataset_and_embedding_args,  # 获取数据集和嵌入参数
    get_embedding_vector_storage_multiplier,  # 获取嵌入向量存储乘数
    maybe_load_ckpts,                # 加载检查点
    train_with_pipeline,             # 使用流水线训练
)

# 导入Megatron训练框架参数解析和验证模块
from megatron.training.arguments import parse_args, validate_args
from megatron.training.yaml_arguments import validate_yaml
from megatron.training.global_vars import set_global_variables


# ============================================
# 章节2: 排序任务参数类定义
# 使用gin-configurable装饰器使该类的属性可以通过gin配置文件设置
# ============================================

@gin.configurable  # 允许通过gin文件配置该数据类的字段 / Allow gin config for this dataclass
@dataclass         # 自动为类生成__init__等方法 / Auto-generate __init__ etc.
class RankingArgs:
    """
    排序任务的参数配置类。
    Ranking task arguments configuration class.
    
    包含预测头架构、激活函数类型、偏置设置、任务数量和评估指标等配置。
    Contains prediction head arch, activation type, bias, num tasks, eval metrics, etc.
    """
    prediction_head_arch: List[int] = cast(List[int], None)  # 预测头（MLP）的层维度列表，如[256, 128, 64, 1]
    prediction_head_act_type: str = "relu"  # 预测头激活函数类型，支持'relu'或'gelu'
    prediction_head_bias: bool = True  # 预测头是否使用偏置项
    num_tasks: int = 1  # 多任务学习的任务数量，默认为单任务
    eval_metrics: Tuple[str, ...] = ("AUC",)  # 评估指标元组，如("AUC", "LogLoss")

    def __post_init__(self):
        """
        数据类初始化后的验证逻辑。
        确保prediction_head_arch必须提供，且激活函数类型合法。
        """
        assert (
            self.prediction_head_arch is not None
        ), "Please provide prediction head arch for ranking model"
        if isinstance(self.prediction_head_act_type, str):
            # 验证激活函数类型是否合法
            assert self.prediction_head_act_type.lower() in [
                "relu",
                "gelu",
            ], "prediction_head_act_type should be in ['relu', 'gelu']"


# ============================================
# 章节3: 配置创建函数
# 将各个参数组装成统一的RankingConfig配置对象
# ============================================

def create_ranking_config(dataset_args: DatasetArgs, network_args: NetworkArgs, embedding_args: EmbeddingArgs) -> RankingConfig:
    """
    创建排序任务的完整配置。
    Create complete ranking task config.
    
    参数/Args:
        dataset_args: 数据集相关参数 / dataset args
        network_args: 网络架构相关参数 / network args
        embedding_args: 嵌入层相关参数 / embedding args
        
    返回/Returns:
        RankingConfig: 完整的排序任务配置对象 / complete ranking config object
    """
    ranking_args = RankingArgs()  # 创建排序参数实例

    return RankingConfig(
        # 创建嵌入层配置
        embedding_configs=create_embedding_configs(
            dataset_args, network_args, embedding_args
        ),
        # 从RankingArgs复制各项配置
        prediction_head_arch=ranking_args.prediction_head_arch,
        prediction_head_act_type=ranking_args.prediction_head_act_type,
        prediction_head_bias=ranking_args.prediction_head_bias,
        num_tasks=ranking_args.num_tasks,
        eval_metrics=ranking_args.eval_metrics,
    )


# ============================================
# 章节4: 命令行参数解析扩展
# ============================================

def extra_init_args(parser: argparse.ArgumentParser):
    """
    添加额外的命令行参数到Megatron的解析器。
    Add extra command line args to Megatron's parser.
    
    主要添加gin配置文件路径参数。
    Mainly adds gin config file path argument.
    """
    group = parser.add_argument_group('Distributed GR Arguments')  # 创建参数组
    group.add_argument("--gin-config-file", type=str)  # gin配置文件路径
    return parser


# ============================================
# 章节5: 主函数 - 训练流程入口
# ============================================

def main():
    # 检查是否启用FSDP2（Fully Sharded Data Parallel 2）
    # 通过环境变量USE_FSDP2控制，默认为关闭
    use_fsdp2 = os.getenv('USE_FSDP2', '0').lower() in ('1', 'true')

    # 解析Megatron命令行参数，传入额外的参数解析器
    args = parse_args(extra_init_args, True)
    
    # 如果使用FSDP2，需要验证YAML配置或标准参数
    if use_fsdp2:
        if args.yaml_cfg is not None:
            args = validate_yaml(args, {})  # 验证YAML配置
        else:
            validate_args(args, {})  # 验证标准参数
        set_global_variables(args, False)  # 设置Megatron全局变量

    # 创建独立的参数解析器用于gin配置
    parser = argparse.ArgumentParser(
        description="Distributed GR Arguments", allow_abbrev=False
    )
    parser.add_argument("--gin-config-file", type=str)
    
    # 解析gin配置文件，动态设置各配置类参数
    gin.parse_config_file(args.gin_config_file)
    
    # 通过gin配置创建各参数类的实例
    trainer_args = TrainerArgs()           # 训练参数
    dataset_args, embedding_args = get_dataset_and_embedding_args()  # 数据集和嵌入参数
    network_args = NetworkArgs()           # 网络参数
    optimizer_args = OptimizerArgs()     # 优化器参数
    tp_args = TensorModelParallelArgs()    # 张量并行参数

    # ============================================
    # 步骤1: 初始化分布式环境
    # ============================================
    init.initialize_distributed()  # 初始化分布式通信
    init.initialize_model_parallel(
        tensor_model_parallel_size=tp_args.tensor_model_parallel_size  # 设置TP大小
    )
    init.set_random_seed(trainer_args.seed)  # 设置随机种子，确保可复现性
    
    # 打印NPU内存信息
    free_memory, total_memory = torch_npu.npu.mem_get_info()
    print_rank_0(
        f"distributed env initialization done. Free cuda memory: {free_memory / (1024 ** 2):.2f} MB"
    )

    # ============================================
    # 步骤2: 创建模型配置和实例化模型
    # ============================================
    hstu_config = create_hstu_config(network_args, tp_args)  # 创建HSTU配置
    task_config = create_ranking_config(dataset_args, network_args, embedding_args)  # 创建任务配置
    model = get_ranking_model(hstu_config=hstu_config, task_config=task_config)  # 实例化排序模型

    # 创建动态嵌入选项字典
    dynamic_options_dict = create_dynamic_optitons_dict(
        embedding_args,
        network_args.hidden_size,
        training=True,
        embedding_dim_multiplier=get_embedding_vector_storage_multiplier(
            optimizer_args.optimizer_str
        ),
    )

    # 创建优化器参数
    optimizer_param = create_optimizer_params(optimizer_args)

    # ============================================
    # 步骤3: 模型分片和优化器创建
    # 根据是否使用FSDP2选择不同的分片策略
    # ============================================
    if use_fsdp2:
        # 使用FSDP2进行模型分片
        print_rank_0("Using FSDP2 for model sharding")
        if network_args.dtype_str == "bfloat16":
            model.bfloat16()  # 转换为bfloat16精度
        # 调用FSDP2分片函数，包装模型并创建优化器
        model_train, dense_optimizer = make_optimizer_and_shard_fsdp2(
            model,
            config=hstu_config,
            sparse_optimizer_param=optimizer_param,
            dense_optimizer_param=optimizer_param,
            dynamicemb_options_dict=dynamic_options_dict,
            pipeline_type=trainer_args.pipeline_type,
        )
    else:
        # 使用标准分片策略
        print_rank_0(f"Using standard sharding with pipeline_type: {trainer_args.pipeline_type}")
        model_train, dense_optimizer = make_optimizer_and_shard(
            model,
            config=hstu_config,
            sparse_optimizer_param=optimizer_param,
            dense_optimizer_param=optimizer_param,
            dynamicemb_options_dict=dynamic_options_dict,
            pipeline_type=trainer_args.pipeline_type,
        )

    # ============================================
    # 步骤4: 创建评估指标模块
    # ============================================
    stateful_metric_module = get_multi_event_metric_module(
        num_classes=task_config.prediction_head_arch[-1],  # 预测头最后一层维度作为类别数
        num_tasks=task_config.num_tasks,                    # 任务数量
        metric_types=task_config.eval_metrics,              # 评估指标类型
        comm_pg=parallel_state.get_data_parallel_group(
            with_context_parallel=True
        ),  # 同一TP组的rank执行相同计算
    )

    # ============================================
    # 步骤5: 创建数据加载器
    # ============================================
    train_dataloader, test_dataloader = get_data_loader(
        "ranking", dataset_args, trainer_args, task_config.num_tasks
    )
    
    # 再次打印NPU内存信息，观察模型初始化后的内存变化
    free_memory, total_memory = torch_npu.npu.mem_get_info()
    print_rank_0(
        f"model initialization done, start training. Free cuda memory: {free_memory / (1024 ** 2):.2f} MB"
    )

    # ============================================
    # 步骤6: 加载检查点（如有）
    # ============================================
    maybe_load_ckpts(trainer_args.ckpt_load_dir, model, dense_optimizer)

    # ============================================
    # 步骤7: 创建训练流水线
    # 根据pipeline_type选择不同的流水线实现
    # ============================================
    if trainer_args.pipeline_type in ["prefetch", "native"]:
        if trainer_args.pipeline_type == "prefetch":
            # 创建Prefetch流水线，支持prefetch重叠模式
            pipeline = JaggedMegatronPrefetchTrainPipelineSparseDist(
                model_train,
                dense_optimizer,
                device=torch.device("npu", torch_npu.npu.current_device()),
                prefetch_overlap_mode=trainer_args.prefetch_overlap_mode,
                prefetch_debug=trainer_args.prefetch_debug,
                prefetch_debug_interval=trainer_args.prefetch_debug_interval,
            )
        else:
            # 创建标准Megatron流水线
            pipeline = JaggedMegatronTrainPipelineSparseDist(
                model_train,
                dense_optimizer,
                device=torch.device("npu", torch_npu.npu.current_device()),
            )
    else:
        # 无流水线模式
        pipeline = JaggedMegatronTrainNonePipeline(
            model_train,
            dense_optimizer,
            device=torch.device("npu", torch_npu.npu.current_device()),
        )

    # ============================================
    # 步骤8: 启动训练
    # ============================================
    train_with_pipeline(
        pipeline,
        stateful_metric_module,
        trainer_args,
        train_dataloader,
        test_dataloader,
        dense_optimizer,
    )
    
    # 训练结束，销毁全局状态
    init.destroy_global_state()


if __name__ == "__main__":
    main()
