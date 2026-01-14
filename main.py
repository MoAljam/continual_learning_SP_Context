import os
import json
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import MLP_SP
from train import *
from analysis import *

PLOTS_DIR = "./plots"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# data
BATCH = 128

# experiment
N_TASKS = 10
BLOCKS = 2
EPOCHS_PER_TASK = 1

# models
IN_DIM = 28 * 28
OUT_DIM = 10
HL_DIMS = [128, 64, 64]
CONTEXT_LAYERS_MASK = [1, 1, 1, 1]  # first entrie is for input context
# HL_DIMS = [128, 128]
# CONTEXT_LAYERS_MASK = [1, 1, 1]  # first entrie is for input context

# training
LR = 1e-3

# checkpoints / later analysis
SAVE_ON_TASKS = [0, N_TASKS // 2, N_TASKS - 1]
CHECKPOINT_DIR = "./models_checkpoints"

# seed
SEED = 10

# seed numpy and torch
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def save_results(trace, filename):
    global CHECKPOINT_DIR
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    filename = os.path.join(CHECKPOINT_DIR, filename)
    with open(filename, "w") as f:
        json.dump(trace, f)
    return filename


if __name__ == "__main__":
    eval_on_tasks = list(range(N_TASKS))  # which tasks to eval on during training

    os.makedirs(PLOTS_DIR, exist_ok=True)
    config = {
        "input_dim": IN_DIM,
        "output_dim": OUT_DIM,
        "hl_dims": HL_DIMS,
        "n_tasks": N_TASKS,
        "device": str(DEVICE),
        "context_layers_mask": CONTEXT_LAYERS_MASK,
    }
    metadata = {
        "batch_size": BATCH,
        "epochs_per_task": EPOCHS_PER_TASK,
        "blocks": BLOCKS,
        "lr": LR,
        "seed": SEED,
    }

    # partial class constroctor for models
    MLP_SP_exp = partial(
        MLP_SP,
        **config,
    )

    # criterion
    criterion = F.cross_entropy

    # data loaders
    train_loader, test_loader = init_dataloader_mnist(BATCH, shuffle=True)

    # permutations for different tasks
    input_permutations = [np.random.permutation(28 * 28) for _ in range(N_TASKS)]

    # set identity permutation for first task
    input_permutations[0] = np.arange(28 * 28)  # identity for first task
    input_permutations = np.array(input_permutations)

    # save permutations to file
    np.save("mnist_permutations.npy", np.array(input_permutations))
    # not necessary, but for convenience, keep everything in torch tensor on DEVICE
    # permutation_tensors = torch.tensor(input_permutations, dtype=torch.long, device=DEVICE)

    # partial for run_experiment
    run_experiment = partial(
        run_experiment,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        input_permutations=input_permutations,
        n_tasks=N_TASKS,
        blocks=BLOCKS,
        epochs_per_task=EPOCHS_PER_TASK,
        device=DEVICE,
        save_on_tasks=SAVE_ON_TASKS,
        save_dir=CHECKPOINT_DIR,
        eval_on_tasks=eval_on_tasks,
    )
    # run for baseline model
    model_baseline = MLP_SP_exp(use_context=False, use_task_ro=False)
    optimizer = optim.Adam(model_baseline.parameters(), lr=LR)
    trace_baseline = run_experiment(model_baseline, optimizer)
    trace_baseline["metadata"] = {
        "model_type": "baseline",
        **metadata,
    }
    trace_baseline["model_config"] = {
        **config,
        "use_context": False,
        "use_task_ro": False,
    }

    c_m_str = "_".join([str(n) for n in CONTEXT_LAYERS_MASK])
    save_results(trace_baseline, f"trace_b_cm_{c_m_str}.json")

    # run for context model
    model_context = MLP_SP_exp(use_context=True, use_task_ro=False)
    optimizer = optim.Adam(model_context.parameters(), lr=LR)
    trace_contex = run_experiment(model_context, optimizer)
    trace_contex["metadata"] = {
        "model_type": "context",
        **metadata,
    }
    trace_contex["model_config"] = {
        **config,
        "use_context": True,
        "use_task_ro": False,
    }
    save_results(trace_contex, f"trace_c_cm_{c_m_str}.json")

    # run for baseline model with task-specific readout
    model_baseline_tro = MLP_SP_exp(use_context=False, use_task_ro=True)
    optimizer = optim.Adam(model_baseline_tro.parameters(), lr=LR)
    trace_baseline_tro = run_experiment(model_baseline_tro, optimizer)
    trace_baseline_tro["metadata"] = {
        "model_type": "baseline_tro",
        **metadata,
    }
    trace_baseline_tro["model_config"] = {
        **config,
        "use_context": False,
        "use_task_ro": True,
    }
    save_results(trace_baseline_tro, f"trace_b_tro_cm_{c_m_str}.json")

    # run for context model with task-specific readout
    model_context_tro = MLP_SP_exp(use_context=True, use_task_ro=True)
    optimizer = optim.Adam(model_context_tro.parameters(), lr=LR)
    trace_contex_tro = run_experiment(model_context_tro, optimizer)
    trace_contex_tro["metadata"] = {
        "model_type": "context_tro",
        **config,
    }
    trace_contex_tro["model_config"] = {
        **config,
        "use_context": True,
        "use_task_ro": True,
    }
    save_results(trace_contex_tro, f"trace_c_tro_cm_{c_m_str}.json")
