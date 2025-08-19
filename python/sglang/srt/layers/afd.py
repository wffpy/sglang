# AF disaggregation

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========

import itertools
from collections import deque
from enum import Enum, auto
from typing import Any, Dict, Optional, List, Tuple
from abc import ABC, abstractmethod
from functools import cache

import torch
from torch import nn
import torch.distributed as dist
import zmq

from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.layers.communicator import LayerCommunicator, ScatterMode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.afd_type import AFDPerspective

class AFDForwardStage(Enum):
    AFD_FORWARD_STAGE_A = auto()
    AFD_FORWARD_STAGE_F = auto()

class AFDStageScheduleGenerator:
    Schedule = List[Tuple[AFDForwardStage, int, int]]
    @staticmethod
    def ffn_stage(num_layers: int, m_stage: int) -> Schedule:
        schedule = []
        for l, m in itertools.product(range(num_layers), range(m_stage)):
            schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_A, l, m))
            schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_F, l, m))
        return schedule
    @staticmethod
    def attn_stage(num_layers: int, m_stage: int) -> Schedule:
        schedule = []
        if num_layers == 1:
            return (
                [(AFDForwardStage.AFD_FORWARD_STAGE_A, 0, m) for m in range(m_stage)]
                +
                [(AFDForwardStage.AFD_FORWARD_STAGE_F, 0, m) for m in range(m_stage)]
            )
        for l, m in itertools.product(range(num_layers + 1), range(m_stage)):
            if l > 0:
                schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_F, l - 1, m))
            if l < num_layers:
                schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_A, l, m))
        return schedule

class FifoTensorCommunicator(ABC):
    @abstractmethod
    def recv_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def send_tensor(self, x: torch.Tensor):
        raise NotImplementedError

class ZMQSimpleTensorCommunicator(FifoTensorCommunicator):
    def __init__(self, afd_perspective: AFDPerspective):
        super().__init__()
        self.zmq_context = zmq.Context()

        self.start_lport = (
            self.get_ffn_port() if afd_perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN
            else self.get_attn_port()
        )

        self.start_dport = (
            self.get_attn_port() if afd_perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN
            else self.get_ffn_port()
        )

    def get_ffn_port(self) -> int:
        return 40000

    def get_attn_port(self) -> int:
        return 50000

    def get_lport(self) -> int:
        return self.start_lport + 1 + dist.get_rank()

    def get_dport(self) -> int:
        return self.start_dport + 1 + dist.get_rank()

    def get_current_cuda_device(self) -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot get current CUDA device.")
        device_index = torch.cuda.current_device()
        return torch.device(f"cuda:{device_index}")

    @cache
    def get_push_socket(self) -> zmq.Socket:
        socket = self.zmq_context.socket(zmq.PUSH)
        socket.connect(f"tcp://localhost:{self.get_dport()}")
        return socket

    @cache
    def get_pull_socket(self) -> zmq.Socket:
        socket = self.zmq_context.socket(zmq.PULL)
        socket.bind(f"tcp://*:{self.get_lport()}")
        return socket

    def recv_tensor(self) -> torch.Tensor:
        socket = self.get_pull_socket()
        x = socket.recv_pyobj()
        assert isinstance(x, torch.Tensor)
        return x.to(self.get_current_cuda_device())

    def send_tensor(self, x: torch.Tensor):
        socket = self.get_push_socket()
        socket.send_pyobj(x)

@cache
def get_tensor_communicator() -> FifoTensorCommunicator:
    afd_perspective = get_afd_perspective()
    if afd_perspective is not None:
        return ZMQSimpleTensorCommunicator(afd_perspective)
    else:
        raise NotImplementedError

def get_afd_mirco_batch() -> int:
    afd_mirco_batch = global_server_args_dict.get("afd_mirco_batch")
    return afd_mirco_batch

def get_afd_perspective() -> Optional[AFDPerspective]:
    afd_perspective = global_server_args_dict.get("afd_perspective")
    return afd_perspective

def afd_is_ffn():
    return get_afd_perspective() == AFDPerspective.AFD_PERSPECTIVE_FFN

def afd_is_attn():
    return get_afd_perspective() == AFDPerspective.AFD_PERSPECTIVE_ATTN

def model_forward_afd_split_inputs(
    layers,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_data_scatter_mode: ScatterMode,
):
    raise NotImplementedError

def model_forward_afd(
    layers,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    input_data_scatter_mode: ScatterMode,
):
    num_layers = len(layers)
    m_stage = get_afd_mirco_batch()

    input_arrs = model_forward_afd_split_inputs(
        layers=layers,
        hidden_states=hidden_states,
        residual=residual,
        positions=positions,
        forward_batch=forward_batch,
        input_data_scatter_mode=input_data_scatter_mode
    )

    stage_outputs: Dict[AFDForwardStage, deque[dict[Any, Any]]] = {
        AFDForwardStage.AFD_FORWARD_STAGE_A: deque(),
        AFDForwardStage.AFD_FORWARD_STAGE_F: deque(),
    }

    stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].extend(input_arrs)

    def forward_A(layer_id: int, mirco_batch_idx: int):
        inputs_args = stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].popleft()
        hidden_states, residual = layers[layer_id].forward_afd_A(
            input_arrs[mirco_batch_idx]["positions"],
            inputs_args["hidden_states"],
            input_arrs[mirco_batch_idx]["forward_batch"],
            inputs_args["residual"],
        )
        stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_A].append(
            dict (
            hidden_states = hidden_states,
            residual = residual,
        ))

    def forward_F(layer_id: int, mirco_batch_idx: int):
        inputs_args = stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_A].popleft()
        hidden_states, residual = layers[layer_id].forward_afd_F(
            inputs_args["hidden_states"],
            input_arrs[mirco_batch_idx]["forward_batch"],
            inputs_args["residual"],
        )
        stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].append(
            dict (
            hidden_states = hidden_states,
            residual = residual,
        ))

    stage_executors = {
        AFDForwardStage.AFD_FORWARD_STAGE_A : forward_A,
        AFDForwardStage.AFD_FORWARD_STAGE_F : forward_F,
    }

    pipeline_stages = (
        AFDStageScheduleGenerator.attn_stage(num_layers, m_stage)
        if afd_is_attn()
        else AFDStageScheduleGenerator.ffn_stage(num_layers, m_stage)
    )

    for stage in pipeline_stages:
        type, *args = stage
        stage_executors.get(type)(*args)

    try:
        results = [stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].popleft() for _ in range(m_stage)]
    except IndexError:
        raise ValueError("model_forward_afd: impossible path, a potential implementation bug?")

    all_hidden_states, all_residual = zip(
        *((res["hidden_states"], res["residual"]) for res in results)
    )

    return (
        torch.cat(all_hidden_states, dim=0),
        torch.cat(all_residual, dim=0) if afd_is_attn() else None,
    )

class AFDCommunicator(LayerCommunicator):
    def __init__(self, layer_communicator: LayerCommunicator, perspective: AFDPerspective, layer_id: int):
        self.perspective = perspective
        self.layer_communicator = layer_communicator
        self.layer_id = layer_id
        pass
    def prepare_attn(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        # just pass through
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
            return hidden_states, residual

        return self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)

    def prepare_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
            hidden_states = get_tensor_communicator().recv_tensor()
            return hidden_states, residual

        hidden_states, residual = self.layer_communicator.prepare_mlp(hidden_states, residual, forward_batch)
        get_tensor_communicator().send_tensor(hidden_states)

        return hidden_states, residual

    def postprocess_layer(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if self.perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
            get_tensor_communicator().send_tensor(hidden_states)
            return hidden_states, residual

        hidden_states = get_tensor_communicator().recv_tensor()

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual

class AFDProxyAttention(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch) -> torch.Tensor:
        return hidden_states

class AFDProxyMLP(nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                forward_batch: Optional[ForwardBatch] = None) -> torch.Tensor:
        return hidden_states
