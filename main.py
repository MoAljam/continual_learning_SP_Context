import os
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
BLOCKS = 1
EPOCHS_PER_TASK = 1

# models
IN_DIM = 28 * 28
OUT_DIM = 10
HL_DIMS = [128, 128]
CONTEXT_LAYERS_MASK = [1, 1, 1]  # first entrie is for input context

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


def add_boundary_lines(axs, B, T, E):
    task_lines = np.arange(0, B * T * E + 1, E)[1:-1]
    block_lines = np.arange(0, B * T * E + 1, T * E)[1:-1]
    for ax in axs:
        for epoch in task_lines:
            ax.axvline(x=epoch, color="gray", linestyle="--", linewidth=1, alpha=0.2)
        for epoch in block_lines:
            ax.axvline(x=epoch, color="black", linestyle="--", linewidth=2, alpha=0.4)


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # partial class constroctor for models
    MLP_SP_exp = partial(
        MLP_SP,
        input_dim=IN_DIM,
        output_dim=OUT_DIM,
        hl_dims=HL_DIMS,
        num_tasks=N_TASKS,
        device=DEVICE,
        context_layers_mask=CONTEXT_LAYERS_MASK,
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
    )
    # run for baseline model
    model_baseline = MLP_SP_exp(use_context=False, use_task_ro=False)
    optimizer = optim.Adam(model_baseline.parameters(), lr=LR)
    trace_baseline = run_experiment(model_baseline, optimizer)

    # run for context model
    model_context = MLP_SP_exp(use_context=True, use_task_ro=False)
    optimizer = optim.Adam(model_context.parameters(), lr=LR)
    trace_contex = run_experiment(model_context, optimizer)

    # run for baseline model with task-specific readout
    model_baseline_tro = MLP_SP_exp(use_context=False, use_task_ro=True)
    optimizer = optim.Adam(model_baseline_tro.parameters(), lr=LR)
    trace_baseline_tro = run_experiment(model_baseline_tro, optimizer)

    # run for context model with task-specific readout
    model_context_tro = MLP_SP_exp(use_context=True, use_task_ro=True)
    optimizer = optim.Adam(model_context_tro.parameters(), lr=LR)
    trace_contex_tro = run_experiment(model_context_tro, optimizer)

    # region plotting
    # plot results: accuracy on task 0
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs = axs.flatten()

    plot_task_accuracies(trace_contex["accuracy_task_0"], "Context Model", step=EPOCHS_PER_TASK, ax=axs[0])
    plot_task_accuracies(trace_baseline["accuracy_task_0"], "Baseline Model", step=EPOCHS_PER_TASK, ax=axs[0])

    plot_task_accuracies(trace_contex_tro["accuracy_task_0"], "Context Model TRO", step=EPOCHS_PER_TASK, ax=axs[1])
    plot_task_accuracies(trace_baseline_tro["accuracy_task_0"], "Baseline Model TRO", step=EPOCHS_PER_TASK, ax=axs[1])

    axs[0].set_title("CL: (Task 0) same readout")
    axs[1].set_title("CL: (Task 0) task-specific readout")

    axs[0].legend()
    axs[1].legend()

    add_boundary_lines(axs, BLOCKS, N_TASKS, EPOCHS_PER_TASK)

    plt.tight_layout()
    if PLOTS_DIR:
        plt.savefig(os.path.join(PLOTS_DIR, "cl_performance_task_0.png"))
    plt.show()

    # plot results: loss and accuracy on current task to ensure the models are learning properly on the new tasks
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs = axs.flatten()

    plot_task_accuracies(trace_contex["accuracy_task_curr"], "Context Model", step=EPOCHS_PER_TASK, ax=axs[0])
    plot_task_accuracies(
        trace_baseline_tro["accuracy_task_curr"], "Baseline Model TRO", step=EPOCHS_PER_TASK, ax=axs[0]
    )
    plot_task_accuracies(trace_contex_tro["accuracy_task_curr"], "Context Model TRO", step=EPOCHS_PER_TASK, ax=axs[0])
    plot_task_accuracies(trace_baseline["accuracy_task_curr"], "Baseline Model", step=EPOCHS_PER_TASK, ax=axs[0])

    plot_training_loss(trace_contex["loss"], "Context Model", ax=axs[1])
    plot_training_loss(trace_contex_tro["loss"], "Context Model TRO", ax=axs[1])
    plot_training_loss(trace_baseline["loss"], "Baseline Model", ax=axs[1])
    plot_training_loss(trace_baseline_tro["loss"], "Baseline Model TRO", ax=axs[1])

    axs[0].set_title("CL: (current task)")
    axs[1].set_title("Training loss")

    # add vertical lines at block and task boundaries
    add_boundary_lines(axs, BLOCKS, N_TASKS, EPOCHS_PER_TASK)

    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize="small", frameon=False, bbox_to_anchor=(0.5, 1.1))
    plt.tight_layout()
    if PLOTS_DIR:
        plt.savefig(os.path.join(PLOTS_DIR, "cl_performance_task_curr.png"))
    plt.show()
