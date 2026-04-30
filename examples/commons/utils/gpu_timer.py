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
import statistics
from collections import defaultdict
import time

import torch


class GPUTimer:
    def __init__(self):
        if hasattr(torch, "npu") and torch.npu.is_available():
            self._device_module = torch.npu
            self._use_event_timer = True
        elif torch.cuda.is_available():
            self._device_module = torch.cuda
            self._use_event_timer = True
        else:
            self._device_module = None
            self._use_event_timer = False

        if self._use_event_timer:
            self.start_event = self._device_module.Event(enable_timing=True)
            self.end_event = self._device_module.Event(enable_timing=True)
        else:
            self._start_time = 0.0
            self._end_time = 0.0

    def start(self):
        if self._use_event_timer:
            self.start_event.record()
        else:
            self._start_time = time.perf_counter()

    def stop(self):
        if self._use_event_timer:
            self.end_event.record()
        else:
            self._end_time = time.perf_counter()

    def elapsed_time(self):
        """
        return in ms
        """
        if self._use_event_timer:
            self.dist_sync()
            self._device_module.synchronize()
            return self.start_event.elapsed_time(self.end_event)
        return (self._end_time - self._start_time) * 1000.0

    def dist_sync(self):
        if not torch.distributed.is_initialized():
            return
        if self._use_event_timer:
            torch.distributed.barrier(device_ids=[self._device_module.current_device()])
        else:
            torch.distributed.barrier()


class IGPUTimer(GPUTimer):
    def __init__(self, max_iters=1):
        super().__init__()
        self._max_iters = max_iters

        if self._use_event_timer:
            self.start_events = [
                self._device_module.Event(enable_timing=True) for i in range(max_iters)
            ]
            self.end_events = [
                self._device_module.Event(enable_timing=True) for i in range(max_iters)
            ]
        else:
            self._start_times = [0.0 for _ in range(max_iters)]
            self._end_times = [0.0 for _ in range(max_iters)]

        self._recorded_start_events = defaultdict()
        self._recorded_end_events = defaultdict()

    def start(self, ith=0):
        if self._use_event_timer:
            self.start_events[ith].record()
        else:
            self._start_times[ith] = time.perf_counter()
        self._recorded_start_events[ith] = True

    def stop(self, ith=0):
        if self._use_event_timer:
            self.end_events[ith].record()
        else:
            self._end_times[ith] = time.perf_counter()
        self._recorded_end_events[ith] = True

    def elapsed_time(self, reduction="mean"):
        self.dist_sync()
        if self._use_event_timer:
            self._device_module.synchronize()
        times = []
        for idx in self._recorded_start_events.keys():
            assert idx in self._recorded_end_events, "end_event is not recorded "
            if self._use_event_timer:
                times.append(self.start_events[idx].elapsed_time(self.end_events[idx]))
            else:
                times.append((self._end_times[idx] - self._start_times[idx]) * 1000.0)

        self._recorded_start_events.clear()
        self._recorded_end_events.clear()
        if reduction == "mean":
            ret_time = sum(times) / len(times)
        elif reduction == "max":
            ret_time = max(times)
        elif reduction == "median":
            ret_time = statistics.median(times)
        else:
            raise ValueError(f"reduction {reduction} is not supported")

        return ret_time

    def reset(self):
        self._recorded_start_events.clear()
        self._recorded_end_events.clear()

    def dist_sync(self):
        if not torch.distributed.is_initialized():
            return
        if self._use_event_timer:
            torch.distributed.barrier(device_ids=[self._device_module.current_device()])
        else:
            torch.distributed.barrier()
