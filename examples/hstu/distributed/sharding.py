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
# 本文件实现模型分片、优化器创建和分布式数据并行(DDP/FSDP)逻辑
# ============================================

# pyre-strict  # Pyre静态类型检查严格模式
from typing import Any, Dict, List, Set, Tuple, Type, Union, Optional

import torch
import torch_npu
import torch.distributed as dist
import torchrec

# 导入配置类和自定义梯度处理模块
from configs.task_config import OptimizerParam
from distributed.finalize_model_grads import finalize_model_grads

# FBGEMM嵌入配置
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType

# Megatron核心模块
from megatron.core import tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module

# 自定义数据并行嵌入模块
from modules.embedding import DataParallelEmbeddingCollection

# PyTorch分布式优化器
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torch.optim.optimizer import Optimizer

# TorchRec分布式相关模块
from torchrec.distributed.composable.table_batched_embedding_slice import (
    TableBatchedEmbeddingSlice,
)
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    QCommsConfig,
    get_qcomm_codecs_registry,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.types import ShardedTensor, ShardingEnv
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.distributed.embedding_types import ShardingType
from torchrec.distributed.planner import Topology
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig

# 动态嵌入模块（NPU特定实现）
from dynamic_emb import (
    DynamicEmbeddingEnumerator, 
    DynamicEmbParameterConstraints, 
    DynamicEmbTableOptions,
    DynamicEmbeddingShardingPlanner, 
    DynamicEmbeddingCollectionSharder,
)

# 常量定义：数据并行嵌入集合的模块名称
DATA_PARALLEL_EMBEDDING_MODULE_NAME = "_data_parallel_embedding_collection"
from megatron.core import parallel_state

# 导入MindSpeed FSDP实现
from megatron.core.distributed import TorchFullyShardedDataParallel


# ============================================
# 章节2: 自定义BatchFSDP类
# 继承自TorchFullyShardedDataParallel，用于处理批次输入
# ============================================

class BatchFSDP(TorchFullyShardedDataParallel):
    """
    自定义FSDP类，用于处理批次数据。
    重写forward方法，使其直接调用内部module的forward。
    """
    def forward(self, batch):
        return self.module(batch)


# ============================================
# 章节3: 常量定义和映射配置
# 定义TorchRec嵌入类型集合和流水线类型映射
# ============================================

# TorchRec嵌入模块类型集合，用于识别需要分片的模块
TORCHREC_TYPES: Set[Type[Union[EmbeddingBagCollection, EmbeddingCollection]]] = {
    EmbeddingBagCollection,
    EmbeddingCollection,
}

# 流水线类型与计算内核的映射配置（模型并行）
# prefetch模式使用UVM缓存内核，native模式使用标准fused或fused_uvm内核
_pipeline_type_to_model_parallel_allowed_compute_kernels = {
    "prefetch": ["fused_uvm_caching"],  # prefetch流水线使用UVM缓存
    "native": ["fused", "fused_uvm"],  # 原生流水线使用fused或UVM
    "none": [],  # 无流水线模式不限制计算内核
}

# 流水线类型与计算内核的映射配置（数据并行）
# 数据并行通常只使用dense内核
_pipeline_type_to_data_parallel_allowed_compute_kernels = {
    "prefetch": ["dense"],
    "native": ["dense"],
    "none": [],
}

# 分片类型到允许计算内核的映射
_sharding_type_to_allowed_compute_kernels = {
    "data_parallel": _pipeline_type_to_data_parallel_allowed_compute_kernels,
    "model_parallel": _pipeline_type_to_model_parallel_allowed_compute_kernels,
}


# ============================================
# 章节4: 分片规划器创建
# 根据嵌入配置和流水线类型创建分片计划
# ============================================

def get_planner(
    eb_configs: List[EmbeddingConfig],
    data_parallel_embedding_table_names: Set[str],
    dynamicemb_options_dict: Dict[str, DynamicEmbTableOptions],
    device: torch.device,
    pipeline_type: str = "none",
    ddr_cap: int = 512 * 1024 * 1024 * 1024,  # 默认假设节点有512GB内存
    intra_host_bw: int = 450e9,  # NVLink带宽约450GB/s
    inter_host_bw: int = 25e9,   # 网卡带宽约25GB/s
):
    """
    创建分片规划器，为每个嵌入表生成分片约束。
    
    参数:
        eb_configs: 嵌入配置列表
        data_parallel_embedding_table_names: 数据并行嵌入表名称集合
        dynamicemb_options_dict: 动态嵌入选项字典
        device: 目标设备
        pipeline_type: 流水线类型（prefetch/native/none）
        ddr_cap: DDR内存容量
        intra_host_bw: 节点内带宽
        inter_host_bw: 节点间带宽
        
    返回:
        DynamicEmbeddingShardingPlanner: 分片规划器实例
    """
    constraints = {}
    
    # 为每个嵌入配置创建分片约束
    for config in eb_configs:
        if config.name in data_parallel_embedding_table_names:
            # 数据并行嵌入表使用dense计算内核
            compute_kernel_type = _sharding_type_to_allowed_compute_kernels[
                "data_parallel"
            ][pipeline_type]
            constraint = DynamicEmbParameterConstraints(
                sharding_types=[
                    ShardingType.DATA_PARALLEL.value,  # 数据并行分片
                ],
                use_dynamicemb=False,  # 不使用动态嵌入
                compute_kernels=compute_kernel_type,
            )
        elif config.name in dynamicemb_options_dict:
            # 动态嵌入表配置
            compute_kernel_type = ["fused"]  # 使用fused计算内核
            dynamicemb_options = dynamicemb_options_dict[config.name]
            constraint = DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],  # 行级分片
                dynamicemb_options=dynamicemb_options,
                compute_kernels=compute_kernel_type,
            )
        else:
            # 模型并行嵌入表使用对应流水线的计算内核
            compute_kernel_type = _sharding_type_to_allowed_compute_kernels[
                "model_parallel"
            ][pipeline_type]
            # TODO: 保存和加载不支持表级分片，暂时禁用
            constraint = DynamicEmbParameterConstraints(
                sharding_types=[
                    ShardingType.ROW_WISE.value,  # 行级分片
                ],
                use_dynamicemb=False,
                compute_kernels=compute_kernel_type,
            )
        constraints.update({config.name: constraint})

    # 创建拓扑对象，描述分布式环境
    topology = Topology(
        world_size=dist.get_world_size(),  # 全局rank数量
        compute_device=device.type,         # 计算设备类型
    )
    
    # 创建枚举器用于搜索分片方案
    enumerator = DynamicEmbeddingEnumerator(
        topology=topology,
        constraints=constraints,
    )
    
    # 创建并返回分片规划器
    return DynamicEmbeddingShardingPlanner(
        eb_configs=eb_configs,
        topology=topology,
        constraints=constraints,
        enumerator=enumerator,
    )


# ============================================
# 章节5: Megatron DDP应用
# 将模型包装为Megatron的DistributedDataParallel
# ============================================

def apply_megatron_ddp(
    model: Union[DistributedModelParallel, torch.nn.Module],
    config: TransformerConfig,
    dense_optimizer_param: OptimizerParam,
    device: torch.device,
):
    """
    应用Megatron DDP到模型。
    Apply megatron DDP to the model.
    
    如果原始模型是DistributedModelParallel，则包装其_dmp_wrapped_module；
    否则直接包装原始模型。
    If the original model is a DistributedModelParallel, the model._dmp_wrapped_module will be wrapped by DDP.
    Otherwise the original model will be wrapped by DDP.
    
    参数/Args:
        model: 待包装的模型 / model to wrap
        config: Transformer配置 / Transformer config
        dense_optimizer_param: 稠密优化器参数 / dense optimizer params
        device: 目标设备 / target device
        
    返回/Returns:
        Tuple[Union[DistributedModelParallel, DDP], Optimizer]: 
            包装后的模型和稠密优化器 / wrapped model and optimizer
    """
    original_model = model  # 保存原始模型引用
    
    # 如果是DMP包装的模型，获取内部模块
    if isinstance(model, DistributedModelParallel):
        model = original_model._dmp_wrapped_module
    else:
        model = original_model
        
    # 将模型移到目标设备
    model = model.to(device)
    
    # 如果使用混合精度，包装为Float16Module
    if config.fp16 or config.bf16:
        model = Float16Module(config, model)

    # 创建DDP配置
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,       # FP32梯度归约
        overlap_grad_reduce=False,      # 不重叠梯度归约
        use_distributed_optimizer=False, # 不使用分布式优化器
        check_for_nan_in_grad=False,    # 不检查NaN梯度
        bucket_size=True,
    )
    
    # MCORE DDP不会隐式广播参数，需要手动处理
    # 包装模型为DDP
    if isinstance(original_model, DistributedModelParallel):
        original_model._dmp_wrapped_module = DDP(
            config,
            ddp_config,
            model,
        )
    else:
        original_model = DDP(
            config,
            ddp_config,
            model,
        )

    # 为数据并行嵌入模块广播参数
    # 只在数据并行组内广播，TP组的权重使用相同的随机状态初始化
    def broadcast_params_for_non_model_parallel_embedding_modules():
        data_parallel_group = parallel_state.get_data_parallel_group(
            with_context_parallel=True
        )
        for p in model.parameters():
            # TableBatchedEmbeddingSlice是模型并行嵌入参数，不需要广播
            if not isinstance(p, TableBatchedEmbeddingSlice):
                dist.broadcast(
                    p.data,
                    src=torch.distributed.get_global_rank(data_parallel_group, 0),
                    group=data_parallel_group,
                )

    broadcast_params_for_non_model_parallel_embedding_modules()
    
    # 设置梯度处理函数
    config.finalize_model_grads_func = finalize_model_grads

    # 确定参数数据类型
    param_dtype = torch.float32
    if config.bf16:
        param_dtype = torch.bfloat16
    elif config.fp16:
        param_dtype = torch.float16

    # 创建稠密优化器配置
    dense_optimizer_config = OptimizerConfig(
        optimizer=dense_optimizer_param.optimizer_str,
        lr=dense_optimizer_param.learning_rate,
        adam_beta1=dense_optimizer_param.adam_beta1,
        adam_beta2=dense_optimizer_param.adam_beta2,
        adam_eps=dense_optimizer_param.adam_eps,
        params_dtype=param_dtype,
        bf16=config.bf16,
        fp16=config.fp16,
        weight_decay=dense_optimizer_param.weight_decay,
    )
    
    # 创建Megatron优化器
    dense_optimizer = get_megatron_optimizer(
        dense_optimizer_config,
        [
            original_model._dmp_wrapped_module
            if isinstance(original_model, DistributedModelParallel)
            else original_model
        ],
    )
    return original_model, dense_optimizer


# ============================================
# 章节6: 稀疏优化器工厂
# 根据优化器名称创建对应的TorchRec优化器类
# ============================================

# 优化器名称到FBGEMM优化器类型的映射
_optimizer_str_to_optim_type = {
    "adam": EmbOptimType.ADAM,
    "sgd": EmbOptimType.EXACT_SGD,
    "row_wise_adagrad": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
}

def sparse_optimizer_factory_and_class(
    optimizer_name: str,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    momentum: float,
    learning_rate: float,
) -> Tuple[Type[Optimizer], Dict[str, Any]]:
    """
    根据优化器名称创建对应的TorchRec优化器类和参数。
    Create TorchRec optimizer class and params based on optimizer name.
    
    参数/Args:
        optimizer_name: 优化器名称（adam/sgd/row_wise_adagrad）/ optimizer name
        betas: Adam的beta参数 / Adam beta params
        eps: 数值稳定性常数 / numerical stability constant
        weight_decay: 权重衰减 / weight decay
        momentum: 动量（SGD使用）/ momentum (for SGD)
        learning_rate: 学习率 / learning rate
        
    返回/Returns:
        Tuple[Optimizer类, 参数字典]: 优化器类和初始化参数 / optimizer class and kwargs
    """
    kwargs: Dict[str, Any] = {"lr": learning_rate}
    
    if optimizer_name == "adam":
        optimizer_cls = torchrec.optim.Adam
        beta1, beta2 = betas
        kwargs.update(
            {"beta1": beta1, "beta2": beta2, "eps": eps, "weight_decay": weight_decay}
        )
    elif optimizer_name == "sgd":
        optimizer_cls = torchrec.optim.SGD
        kwargs.update({"weight_decay": weight_decay, "momentum": momentum})
    elif optimizer_name == "row_wise_adagrad":
        optimizer_cls = torchrec.optim.RowWiseAdagrad
        beta1, beta2 = betas
        kwargs.update(
            {
                "eps": eps,
                "beta1": beta1,
                "beta2": beta2,
                "weight_decay": weight_decay,
            }
        )
    else:
        raise Exception("Unsupported optimizer!")

    return optimizer_cls, kwargs


# ============================================
# 章节7: 应用分布式模型并行（DMP）
# 这是核心的模型分片逻辑
# ============================================

def apply_dmp(
    model: torch.nn.Module,
    dynamicemb_options_dict: Dict[str, DynamicEmbTableOptions],
    sparse_optimizer_param: OptimizerParam,
    pg: torch.distributed.ProcessGroup,
    device: torch.device,
    pipeline_type: str = "native",
):
    """
    应用分布式模型并行（Distributed Model Parallel）。
    Apply Distributed Model Parallel.
    
    对模型中的嵌入模块进行分片，并创建对应的优化器。
    Shards embedding modules in the model and creates corresponding optimizers.
    
    参数/Args:
        model: 原始模型 / original model
        dynamicemb_options_dict: 动态嵌入选项 / dynamic embedding options
        sparse_optimizer_param: 稀疏优化器参数 / sparse optimizer params
        pg: 进程组 / process group
        device: 目标设备 / target device
        pipeline_type: 流水线类型 / pipeline type
        
    返回/Returns:
        DistributedModelParallel: 分片后的模型 / sharded model
    """
    # 是否启用prefetch流水线
    enable_prefetch_pipeline = pipeline_type == "prefetch"
    
    # 创建稀疏优化器类和参数
    sparse_opt_cls, sparse_opt_args = sparse_optimizer_factory_and_class(
        optimizer_name=sparse_optimizer_param.optimizer_str,
        betas=(sparse_optimizer_param.adam_beta1, sparse_optimizer_param.adam_beta2),
        eps=sparse_optimizer_param.adam_eps,
        weight_decay=0.0,  # 稀疏参数权重衰减设为0
        momentum=0.0,
        learning_rate=sparse_optimizer_param.learning_rate,
    )
    
    # 验证优化器类型
    assert (
        sparse_optimizer_param.optimizer_str in _optimizer_str_to_optim_type
    ), f"embedding optimizer only support {list(_optimizer_str_to_optim_type.keys())}"
    
    # 创建fused参数配置
    fused_params = {
        "optimizer": _optimizer_str_to_optim_type[sparse_optimizer_param.optimizer_str],
        "learning_rate": sparse_optimizer_param.learning_rate,
        "beta1": sparse_optimizer_param.adam_beta1,
        "beta2": sparse_optimizer_param.adam_beta2,
        "eps": sparse_optimizer_param.adam_eps,
        "output_dtype": SparseType.FP32,  # 输出为FP32
        "cache_precision": SparseType.FP32,  # 缓存精度
        "stochastic_rounding": False,  # 不使用随机舍入
        "prefetch_pipeline": enable_prefetch_pipeline,  # 是否启用prefetch流水线
    }
    
    # 收集嵌入配置和数据并行嵌入表名称
    eb_configs = []
    data_parallel_embedding_table_names = []
    data_parallel_embedding_module_names = []
    
    # 遍历模型所有模块，识别TorchRec嵌入模块
    for k, module in model.named_modules():
        if type(module) in TORCHREC_TYPES:
            # 为嵌入参数应用反向优化器
            for _, param in module.named_parameters(prefix=k):
                if param.requires_grad:
                    apply_optimizer_in_backward(
                        sparse_opt_cls, [param], sparse_opt_args
                    )
            # 收集嵌入配置
            eb_configs.extend(module.embedding_configs())
            
            # 识别数据并行嵌入模块
            if DATA_PARALLEL_EMBEDDING_MODULE_NAME in k:
                data_parallel_embedding_module_names.append(k)
                for config in module.embedding_configs():
                    data_parallel_embedding_table_names.append(config.name)

    # 创建分片规划器
    planner = get_planner(
        eb_configs,
        set(data_parallel_embedding_table_names),
        dynamicemb_options_dict,
        device,
        pipeline_type,
    )
    
    # 创建通信编解码器配置（使用FP32精度）
    qcomm_codecs_registry = get_qcomm_codecs_registry(
        qcomms_config=QCommsConfig(
            forward_precision=CommType.FP32,
            backward_precision=CommType.FP32,
        )
    )
    
    # 创建分片器列表
    sharders = [
        # NPU目前仅支持DynamicEmbeddingCollectionSharder接口
        DynamicEmbeddingCollectionSharder(
            qcomm_codecs_registry=qcomm_codecs_registry,
            use_index_dedup=True,  # 使用索引去重
            fused_params=fused_params,
        ),
    ]
    
    # 收集集体分片计划
    plan = planner.collective_plan(model, sharders, pg)
    
    # 移除数据并行分片计划（单独处理）
    data_parallel_sharding_plans = []
    for data_parallel_embedding_module_name in data_parallel_embedding_module_names:
        data_parallel_sharding_plans.append(
            plan.plan.pop(data_parallel_embedding_module_name, None)
        )
    
    # 使用fork的随机状态对模型进行分片
    # 确保不同rank有不同的随机状态
    with tensor_parallel.get_cuda_rng_tracker().fork("sharded-embedding-group-seed"):
        model = DistributedModelParallel(
            module=model,
            device=device,
            sharders=sharders,
            plan=plan,
            init_data_parallel=True,  # 开启数据并行
        )

    # 验证非融合稀疏参数为空
    non_fused_sparse_params = {}
    for k, v in in_backward_optimizer_filter(model.named_parameters()):
        if v.requires_grad:
            if isinstance(v, ShardedTensor):
                non_fused_sparse_params[k] = v
    assert len(non_fused_sparse_params) == 0, "non_fused_sparse_params should be empty"

    # 处理数据并行嵌入模块
    if len(data_parallel_sharding_plans) > 0:
        unwrapped_model = model.module
        for dp_module_name, dp_sharding_plan in zip(
            data_parallel_embedding_module_names, data_parallel_sharding_plans
        ):
            # 获取父模块路径
            data_parallel_embedding_collection_father_module_name = (
                dp_module_name.split(".")[:-1]
            )
            father_module = unwrapped_model
            for name in data_parallel_embedding_collection_father_module_name:
                father_module = getattr(father_module, name)
            
            # 获取原始的数据并行嵌入集合
            data_parallel_embedding_collection = getattr(
                father_module, DATA_PARALLEL_EMBEDDING_MODULE_NAME
            )
            
            # 替换为自定义的数据并行嵌入集合
            setattr(
                father_module,
                DATA_PARALLEL_EMBEDDING_MODULE_NAME,
                DataParallelEmbeddingCollection(
                    data_parallel_embedding_collection,
                    dp_sharding_plan,
                    ShardingEnv.from_process_group(pg),
                    fused_params,
                    device,
                ),
            )
    return model


# ============================================
# 章节8: 创建优化器和分片（标准模式）
# ============================================

def make_optimizer_and_shard(
    model: torch.nn.Module,
    config: TransformerConfig,
    sparse_optimizer_param: OptimizerParam,
    dense_optimizer_param: OptimizerParam,
    dynamicemb_options_dict: Dict[str, DynamicEmbTableOptions] = {},
    pipeline_type: str = "native",
    device: torch.device = None,
    pg: torch.distributed.ProcessGroup = None,
) -> Tuple[DistributedModelParallel, torch.optim.Optimizer]:
    """
    创建优化器并应用模型分片（标准模式）。
    Create optimizer and apply model sharding (standard mode).
    
    流程/Flow:
    1. 应用DMP进行嵌入层分片 / apply DMP for embedding sharding
    2. 应用Megatron DDP进行稠密层并行 / apply Megatron DDP for dense layers
    
    参数/Args:
        model: 原始模型 / original model
        config: Transformer配置 / Transformer config
        sparse_optimizer_param: 稀疏优化器参数 / sparse optimizer params
        dense_optimizer_param: 稠密优化器参数 / dense optimizer params
        dynamicemb_options_dict: 动态嵌入选项 / dynamic embedding options
        pipeline_type: 流水线类型 / pipeline type
        device: 目标设备 / target device
        pg: 进程组 / process group
        
    返回/Returns:
        Tuple[DistributedModelParallel, Optimizer]: 分片后的模型和优化器 / sharded model and optimizer
    """
    # 设置默认值
    if device is None:
        device = torch.device("npu", torch_npu.npu.current_device())
    if pg is None:
        pg = dist.group.WORLD

    # 步骤1: 应用DMP进行嵌入层分片
    model = apply_dmp(
        model,
        dynamicemb_options_dict,
        sparse_optimizer_param,
        pg,
        device,
        pipeline_type,
    )
    
    # 步骤2: 应用Megatron DDP进行稠密层并行
    model, dense_optimizer = apply_megatron_ddp(
        model, config, dense_optimizer_param, device
    )

    return model, dense_optimizer


# ============================================
# 章节9: FSDP2相关实现
# ============================================

def apply_megatron_fsdp2(
    dmp: DistributedModelParallel,
    config: TransformerConfig,
    dense_optimizer_param: OptimizerParam,
    device: torch.device,
):
    """
    应用Megatron FSDP2到模型。
    Apply Megatron FSDP2 to the model.
    
    FSDP2（Fully Sharded Data Parallel 2）是PyTorch的分布式数据并行实现，
    比DDP更节省内存，支持更大的模型。
    FSDP2 is PyTorch's distributed data parallel implementation,
    more memory-efficient than DDP, supports larger models.
    
    参数/Args:
        dmp: DMP包装的模型 / DMP-wrapped model
        config: Transformer配置 / Transformer config
        dense_optimizer_param: 稠密优化器参数 / dense optimizer params
        device: 目标设备 / target device
        
    返回/Returns:
        Tuple[DistributedModelParallel, Optimizer]: FSDP2包装后的模型和优化器 / FSDP2-wrapped model and optimizer
    """
    # 获取DMP内部模块
    model = dmp._dmp_wrapped_module
    model = model.to(device)
    
    # 应用混合精度
    if config.fp16 or config.bf16:
        model = Float16Module(config, model)
    
    # 创建FSDP2配置
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,  # FSDP2要求关闭分布式优化器
        check_for_nan_in_grad=False,
        bucket_size=True,
    )
    
    # 导入需要包装的模块类型
    from modules.hstu_block import HSTUBlock
    from modules.native_hstu_layer import HSTULayer
    
    # 使用BatchFSDP包装模型，只包装HSTU相关模块
    dmp._dmp_wrapped_module = BatchFSDP(
        config,
        ddp_config,
        model,
        sub_modules_to_wrap={HSTUBlock, HSTULayer},  # 只包装HSTU模块
    )  

    # 确定参数数据类型
    param_dtype = torch.float32
    if config.bf16:
        param_dtype = torch.bfloat16
    elif config.fp16:
        param_dtype = torch.float16

    # 创建稠密优化器配置
    dense_optimizer_config = OptimizerConfig(
        optimizer=dense_optimizer_param.optimizer_str,
        lr=dense_optimizer_param.learning_rate,
        adam_beta1=dense_optimizer_param.adam_beta1,
        adam_beta2=dense_optimizer_param.adam_beta2,
        adam_eps=dense_optimizer_param.adam_eps,
        params_dtype=param_dtype,
        bf16=config.bf16,
        fp16=config.fp16,
    )
    
    # 创建Megatron优化器
    dense_optimizer = get_megatron_optimizer(
        dense_optimizer_config, [dmp._dmp_wrapped_module]
    )
    return dmp, dense_optimizer


def make_optimizer_and_shard_fsdp2(
    model: torch.nn.Module,
    config: TransformerConfig,
    sparse_optimizer_param: OptimizerParam,
    dense_optimizer_param: OptimizerParam,
    dynamicemb_options_dict: Dict[str, DynamicEmbTableOptions] = {},
    pipeline_type: str = "native",
    device: torch.device = None,
    pg: torch.distributed.ProcessGroup = None,
) -> Tuple[BatchFSDP, torch.optim.Optimizer]:
    """
    创建FSDP2包装的模型和优化器（基于MindSpeed FSDP2实现）。
    Create FSDP2-wrapped model and optimizer (based on MindSpeed FSDP2).
    
    本函数：
    This function:
    1. 应用DMP处理嵌入层 / applies DMP for embeddings
    2. 应用FSDP2处理稠密层（HSTU层）/ applies FSDP2 for dense layers (HSTU)
    3. 创建Megatron优化器 / creates Megatron optimizer
    
    参数/Args:
        model: 待包装模型 / model to wrap
        config: Transformer配置 / Transformer config
        sparse_optimizer_param: 稀疏优化器参数 / sparse optimizer params
        dense_optimizer_param: 稠密优化器参数 / dense optimizer params
        dynamicemb_options_dict: 动态嵌入选项 / dynamic embedding options
        pipeline_type: 流水线类型 / pipeline type
        device: 目标设备 / target device
        pg: 进程组 / process group
        
    返回/Returns:
        Tuple[BatchFSDP, Optimizer]: FSDP2包装模型和优化器 / FSDP2-wrapped model and optimizer
    """
    # 设置默认值
    if device is None:
        device = torch.device("npu", torch_npu.npu.current_device())
    if pg is None:
        pg = dist.group.WORLD
    
    # 步骤1: 应用DMP进行嵌入层分片
    model = apply_dmp(
        model, dynamicemb_options_dict, sparse_optimizer_param, pg, device, pipeline_type
    )
    
    # 步骤2: 应用FSDP2进行稠密层并行
    model, dense_optimizer = apply_megatron_fsdp2(
        model, config, dense_optimizer_param, device
    )
    
    return model, dense_optimizer
