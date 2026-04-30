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
import warnings
import sysconfig
import os
# Ignore all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
import argparse
from dataclasses import dataclass
from functools import partial  # pylint: disable-unused-import
from typing import List, Tuple, cast

import commons.utils.initialize as init
import gin
import torch  # pylint: disable-unused-import
import torch_npu
# 导入 NPU 自定义算子库
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
import mindspeed.megatron_adaptor
from commons.utils.logger import print_rank_0
from configs import RankingConfig
from distributed.sharding import make_optimizer_and_shard, make_optimizer_and_shard_fsdp2
from megatron.core import parallel_state
from model import get_ranking_model
from modules.metrics import get_multi_event_metric_module
from pipeline.train_pipeline import (
    JaggedMegatronPrefetchTrainPipelineSparseDist,
    JaggedMegatronTrainNonePipeline,
    JaggedMegatronTrainPipelineSparseDist,
)
from training import (
    NetworkArgs,
    DatasetArgs,
    EmbeddingArgs,
    OptimizerArgs,
    TensorModelParallelArgs,
    TrainerArgs,
    create_dynamic_optitons_dict,
    create_embedding_configs,
    create_hstu_config,
    create_optimizer_params,
    get_data_loader,
    get_dataset_and_embedding_args,
    get_embedding_vector_storage_multiplier,
    maybe_load_ckpts,
    train_with_pipeline,
)
# 导入 Megatron 训练框架相关模块与 NPU 自定义算子
from megatron.training.arguments import parse_args, validate_args
from megatron.training.yaml_arguments import validate_yaml
from megatron.training.global_vars import set_global_variables


@gin.configurable
@dataclass
class RankingArgs:
    prediction_head_arch: List[int] = cast(List[int], None)
    prediction_head_act_type: str = "relu"
    prediction_head_bias: bool = True
    num_tasks: int = 1
    eval_metrics: Tuple[str, ...] = ("AUC",)

    def __post_init__(self):
        assert (
            self.prediction_head_arch is not None
        ), "Please provide prediction head arch for ranking model"
        if isinstance(self.prediction_head_act_type, str):
            assert self.prediction_head_act_type.lower() in [
                "relu",
                "gelu",
            ], "prediction_head_act_type should be in ['relu', 'gelu']"


def create_ranking_config(dataset_args: DatasetArgs, network_args: NetworkArgs, embedding_args: EmbeddingArgs) -> RankingConfig:
    ranking_args = RankingArgs()

    return RankingConfig(
        embedding_configs=create_embedding_configs(
            dataset_args, network_args, embedding_args
        ),
        prediction_head_arch=ranking_args.prediction_head_arch,
        prediction_head_act_type=ranking_args.prediction_head_act_type,
        prediction_head_bias=ranking_args.prediction_head_bias,
        num_tasks=ranking_args.num_tasks,
        eval_metrics=ranking_args.eval_metrics,
    )

def extra_init_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Distributed GR Arguments')
    group.add_argument("--gin-config-file", type=str)
    return parser

def main():
        
    # Check if FSDP2 is enabled
    use_fsdp2 = os.getenv('USE_FSDP2', '0').lower() in ('1', 'true')

    args = parse_args(extra_init_args, True)
    if use_fsdp2:
        if args.yaml_cfg is not None:
            args = validate_yaml(args, {})
        else:
            validate_args(args, {})
        set_global_variables(args, False)

    parser = argparse.ArgumentParser(
        description="Distributed GR Arguments", allow_abbrev=False
    )
    parser.add_argument("--gin-config-file", type=str)
    gin.parse_config_file(args.gin_config_file)
    trainer_args = TrainerArgs()
    dataset_args, embedding_args = get_dataset_and_embedding_args()
    network_args = NetworkArgs()
    optimizer_args = OptimizerArgs()
    tp_args = TensorModelParallelArgs()

    init.initialize_distributed()
    init.initialize_model_parallel(
        tensor_model_parallel_size=tp_args.tensor_model_parallel_size
    )
    init.set_random_seed(trainer_args.seed)
    free_memory, total_memory = torch_npu.npu.mem_get_info()
    print_rank_0(
        f"distributed env initialization done. Free cuda memory: {free_memory / (1024 ** 2):.2f} MB"
    )
    hstu_config = create_hstu_config(network_args, tp_args)
    task_config = create_ranking_config(dataset_args, network_args, embedding_args)
    model = get_ranking_model(hstu_config=hstu_config, task_config=task_config)

    dynamic_options_dict = create_dynamic_optitons_dict(
        embedding_args,
        network_args.hidden_size,
        training=True,
        embedding_dim_multiplier=get_embedding_vector_storage_multiplier(
            optimizer_args.optimizer_str
        ),
    )

    optimizer_param = create_optimizer_params(optimizer_args)

    if use_fsdp2:
        print_rank_0("Using FSDP2 for model sharding")
        if network_args.dtype_str == "bfloat16":
            model.bfloat16()
        model_train, dense_optimizer = make_optimizer_and_shard_fsdp2(
            model,
            config=hstu_config,
            sparse_optimizer_param=optimizer_param,
            dense_optimizer_param=optimizer_param,
            dynamicemb_options_dict=dynamic_options_dict,
            pipeline_type=trainer_args.pipeline_type,
        )
    else:
        print_rank_0(f"Using standard sharding with pipeline_type: {trainer_args.pipeline_type}")
        model_train, dense_optimizer = make_optimizer_and_shard(
            model,
            config=hstu_config,
            sparse_optimizer_param=optimizer_param,
            dense_optimizer_param=optimizer_param,
            dynamicemb_options_dict=dynamic_options_dict,
            pipeline_type=trainer_args.pipeline_type,
        )

    stateful_metric_module = get_multi_event_metric_module(
        num_classes=task_config.prediction_head_arch[-1],
        num_tasks=task_config.num_tasks,
        metric_types=task_config.eval_metrics,
        comm_pg=parallel_state.get_data_parallel_group(
            with_context_parallel=True
        ),  # ranks in the same TP group do the same compute
    )

    train_dataloader, test_dataloader = get_data_loader(
        "ranking", dataset_args, trainer_args, task_config.num_tasks
    )
    free_memory, total_memory = torch_npu.npu.mem_get_info()
    print_rank_0(
        f"model initialization done, start training. Free cuda memory: {free_memory / (1024 ** 2):.2f} MB"
    )

    maybe_load_ckpts(trainer_args.ckpt_load_dir, model, dense_optimizer)
    if trainer_args.pipeline_type in ["prefetch", "native"]:
        pipeline_factory = (
            JaggedMegatronPrefetchTrainPipelineSparseDist
            if trainer_args.pipeline_type == "prefetch"
            else JaggedMegatronTrainPipelineSparseDist
        )
        pipeline = pipeline_factory(
            model_train,
            dense_optimizer,
            device=torch.device("cuda", torch_npu.npu.current_device()),
        )
    else:
        pipeline = JaggedMegatronTrainNonePipeline(
            model_train,
            dense_optimizer,
            device=torch.device("cuda", torch_npu.npu.current_device()),
        )
    train_with_pipeline(
        pipeline,
        stateful_metric_module,
        trainer_args,
        train_dataloader,
        test_dataloader,
        dense_optimizer,
    )
    init.destroy_global_state()


if __name__ == "__main__":
    main()
