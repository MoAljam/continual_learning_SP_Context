import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt


class MLP_SP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        device,
        hl_dims: list[int],
        num_tasks=1,
        use_context=True,
        use_task_ro=False,
        context_layers_mask: list[bool] = None,
    ):
        super(MLP_SP, self).__init__()
        self.use_context = use_context
        self.use_task_ro = use_task_ro
        self.num_tasks = num_tasks
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hl_dims = hl_dims
        self.context_layers_mask = context_layers_mask
        self.device = device

        self.h_fcs = nn.ModuleList()
        for i in range(len(hl_dims)):
            if i == 0:
                self.h_fcs.append(nn.Linear(input_dim, hl_dims[0]))
            else:
                self.h_fcs.append(nn.Linear(hl_dims[i - 1], hl_dims[i]))

        self.fc_outs = nn.ModuleList()
        for _ in range(num_tasks):
            self.fc_outs.append(nn.Linear(hl_dims[-1], output_dim))

        if self.context_layers_mask is None:  # default: all hidden and input layers get context
            self.context_layers_mask = np.ones((len(hl_dims) + 1))

        self._set_up_context()

    def forward(self, x, task_id=0, output_id=None):

        input_context = getattr(self, "context_0")[task_id]
        x = x * input_context

        for idx, h_fc in enumerate(self.h_fcs):
            x = F.relu(h_fc(x))
            context = getattr(self, "context_" + str(idx + 1))[task_id]
            x = x * context

        if output_id is None:
            if self.use_task_ro:
                x = self.fc_outs[task_id](x)
            else:
                x = self.fc_outs[0](x)

        elif output_id == "all:":  # convienience to get out of all heads without repeated calls
            outs = []
            for fc_out in self.fc_outs:
                outs.append(fc_out(x))
            stacked_outs = torch.stack(outs, dim=1)  # shape: (batch_size, num_tasks, output_dim)
            x = stacked_outs

        else:
            x = self.fc_outs[output_id](x)

        return x

    def binary_context_vec(self, dim, num_tasks, device):
        context = torch.randint(0, 2, (num_tasks, dim), device=device, dtype=torch.float32)
        context = context * 2 - 1  # convert to -1 and 1
        return context

    def _set_up_context(self):
        # setup context vectors
        if self.use_context:
            for idx, layer_dim in enumerate([self.input_dim] + self.hl_dims):
                if self.context_layers_mask[idx]:
                    context = self.binary_context_vec(layer_dim, self.num_tasks, self.device)
                else:
                    context = torch.ones((self.num_tasks, layer_dim), device=self.device)
                self.register_buffer(f"context_" + str(idx), context)

        else:  # identity context all layers
            for idx, layer_dim in enumerate([self.input_dim] + self.hl_dims):
                context = torch.ones((self.num_tasks, layer_dim), device=self.device)
                self.register_buffer(f"context_" + str(idx), context)
