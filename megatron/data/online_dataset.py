# Copyright (c) 2025, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
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

"""Online dataset."""
from typing import Union, List

import numpy as np
import torch
import torch.utils.data
import socket
import pickle
from megatron.mpu.initialize import get_data_parallel_rank


class OnlineDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_samples,
        seq_length,
        leave_one_out=False,
        data_split="train",
        dataserver_ips: Union[str, List[str]] = "localhost",
        dataserver_ports: Union[int, List[int]] = 10000,
    ):
        self.num_samples = num_samples
        self.global_rank = get_data_parallel_rank()
        self.leave_one_out = leave_one_out
        self.reward_buffer = []
        self.online_batching_data = []
        self.data_split = data_split
        self.seq_length = seq_length
        self.dataserver_ips = dataserver_ips
        self.dataserver_ports = dataserver_ports

    def __len__(self):
        # dummy value since it's decided by the Online Trainer
        return self.num_samples

    def update_online_batches(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if isinstance(self.dataserver_ips, str):
            ipaddr = self.dataserver_ips
        else:
            ipaddr = self.dataserver_ips[self.global_rank]
        if isinstance(self.dataserver_ports, int):
            # simply add over the global rank
            port = self.dataserver_ports
        else:
            # in case we want to use different ports for different ranks, e.g. per machine sampling
            port = self.dataserver_ports[self.global_rank]
        print(f"Connecting to {ipaddr}:{port}")
        s.connect((ipaddr, port))
        s.send(self.data_split.encode())
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
        batch_data = pickle.loads(data)
        s.close()
        print(f"Received {len(batch_data)} samples from the server.")
        for data in batch_data:
            if self.leave_one_out:
                rewards = list()
                for i in range(len(data["rewards"])):
                    rewards.append(
                        data["rewards"][i]
                        - np.mean(
                            [
                                data["rewards"][j]
                                for j in range(len(data["rewards"]))
                                if j != i
                            ]
                        )
                    )
                data["raw_rewards"] = data["rewards"]
                data["rewards"] = rewards
            else:
                moving_average = 0
                if len(self.reward_buffer) > 0:
                    moving_average = np.mean(self.reward_buffer)
                self.reward_buffer.append(np.mean(data["rewards"]))
                if len(self.reward_buffer) > 100:
                    self.reward_buffer.pop(0)
                # For metrics...
                data["raw_rewards"] = data["rewards"]
                data["rewards"] = [r - moving_average for r in data["rewards"]]
            for i in range(len(data["completions"])):
                self.online_batching_data.append(
                    [
                        data["prefix"],
                        data["completions"][i],
                        data["rewards"][i],
                        data["raw_rewards"][i],
                    ]
                )

    def __getitem__(self, idx):
        if len(self.online_batching_data) == 0:
            self.update_online_batches()
        batch = self.online_batching_data.pop(0)
        text = batch[0] + batch[1]
        label = [-100 for _ in batch[0]] + batch[1]
        # +1 because of causal masking
        if len(text) <= self.seq_length:
            text = text + [0] * ((self.seq_length + 1) - len(text))
            label = label + [-100] * ((self.seq_length + 1) - len(label))
        return {
            "text": np.array(text, dtype=np.int64),
            "label": np.array(label, dtype=np.int64),
            "reward": np.array([batch[2]], dtype=np.float32),
            "raw_reward": np.array([batch[3]], dtype=np.float32),
        }
