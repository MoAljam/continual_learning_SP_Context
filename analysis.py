import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


ACTIVATIONS = {}


def get_activation(name):
    def hook(model, input, output):
        ACTIVATIONS[name] = output.detach()

    return hook


def get_activations_for_model(model, task_id, block, input_permutations, device, sub_loaders):
    # register hook to capture activations from first hidden layer
    # get activations from all hidden layers
    activations_from_layers = []
    get_act_id = lambda task_id, block, l_idx: f"task_{task_id}_block_{block}_l{l_idx}"
    layers_idxs = np.arange(len(model.h_fcs) + 1)

    for hl_idx in layers_idxs:
        id = get_act_id(task_id, block, hl_idx)
        # print("hook: ", id)
        if hl_idx == len(layers_idxs) - 1:
            if model.use_task_ro:
                model.fc_outs[task_id].register_forward_hook(get_activation(id))
            else:
                model.fc_outs[0].register_forward_hook(get_activation(id))

        else:
            model.h_fcs[hl_idx].register_forward_hook(get_activation(id))

    # pass some data through the model to get activations
    for label, sub_loader in sub_loaders.items():
        for images, labels in sub_loader:
            images = images.to(device)
            B = images.size(0)
            x = images.view(B, -1)
            x = x[:, input_permutations[task_id]]
            _ = model(x, task_id=task_id)
        # store activations
        for hl_idx in layers_idxs:
            id = get_act_id(task_id, block, hl_idx)
            activations_from_layers.append(
                {
                    "label": label,
                    "task_id": task_id,
                    "block": block,
                    "layer": hl_idx,
                    "activations": ACTIVATIONS.get(id, torch.tensor([])),
                }
            )
    # print(f"## activations of layers: {np.arange(len(model.h_fcs) + 2)}, task: {task_id}, block: {block}")

    return activations_from_layers


def get_activations_pca(activations_array, n_pcs=2):
    pca = PCA(n_components=n_pcs)
    activations_pca = pca.fit_transform(activations_array)
    return activations_pca


# region Plotting
def plot_activations_pca(activations_list, activations_pca, layer_to_analyze, title="", ax=None, show_plot=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # colors and shapes based on unique task_ids and label
    unique_task_ids = list(set(act["task_id"] for act in activations_list))
    unique_labels = list(set(act["label"] for act in activations_list))

    colors = {task_id: plt.cm.get_cmap("tab10")(i) for i, task_id in enumerate(unique_task_ids)}
    shapes = {label: marker for label, marker in zip(unique_labels, ["o", "P", "D", "^", "s", "v", "*", "X", "<", ">"])}
    # colors = {label: plt.cm.get_cmap('tab10')(i) for i, label in enumerate(unique_labels)}
    # shapes = {task_id: marker for task_id , marker in zip(unique_task_ids, ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>'])}

    start_idx = 0
    for act in activations_list:
        num_samples = act["activations"].size(0)
        task_id = act["task_id"]
        label = act["label"]
        end_idx = start_idx + num_samples
        ax.scatter(
            activations_pca[start_idx:end_idx, 0],
            activations_pca[start_idx:end_idx, 1],
            color=colors[task_id],
            marker=shapes[label],
            # c=colors[label],
            # marker=shapes[task_id],
            label=f"Task {task_id}, Label {label}",
            alpha=0.95,
        )
        start_idx = end_idx
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title(f"{title}\nLayer {layer_to_analyze} activations")
    ax.grid(True)

    if show_plot:
        plt.tight_layout()
        plt.draw()

    return ax


def plot_training_loss(loss, model_name, ax=None, show_plot=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(loss, label=f"{model_name}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss over Epochs")

    if show_plot:
        plt.tight_layout()
        plt.draw()

    return ax


def plot_task_accuracies(acc, model_name, step=1, title="", ax=None, show_plot=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(0, len(acc) * step, step)
    ax.plot(x, acc, label=f"{model_name}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)

    if show_plot:
        plt.tight_layout()
        plt.draw()

    return ax
