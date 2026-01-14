import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from models import MLP_SP
from train import *
from main import (
    DEVICE,
    IN_DIM,
    OUT_DIM,
    HL_DIMS,
    N_TASKS,
    EPOCHS_PER_TASK,
    BATCH,
    LR,
    SAVE_ON_TASKS,
    CHECKPOINT_DIR,
    SEED,
    CONTEXT_LAYERS_MASK,
    BLOCKS,
)

ACTIVATIONS = {}


def load_model(model_type, task_id, device, context_layers_mask=None, block=None):
    cm = CONTEXT_LAYERS_MASK if context_layers_mask is None else context_layers_mask
    cm = "_".join([str(n) for n in cm])
    block = BLOCKS - 1 if block is None else block

    path = os.path.join(CHECKPOINT_DIR, f"model_{model_type}_task_{task_id}_block_{BLOCKS - 1}_cm_{cm}.pth")
    state = torch.load(path, map_location=device)
    model_info_path = os.path.join(CHECKPOINT_DIR, f"trace_{model_type}_cm_{cm}.json")
    with open(model_info_path, "r") as f:
        model_info = json.load(f)

    model_config = model_info["model_config"]
    model_config.update(
        {
            "device": device,
        }
    )
    model_loaded = MLP_SP(
        **model_config,
    )
    model_loaded.load_state_dict(state)
    model_loaded.to(device)
    return model_loaded


# region activations
def get_activation(name):
    def hook(model, input, output):
        ACTIVATIONS[name] = output.detach()

    return hook


def get_activations_for_model(model, model_at_task, block, target_task, input_permutations, sub_loaders, device):
    # register hook to capture activations from first hidden layer
    # get activations from all hidden layers
    activations_from_layers = []
    get_act_id = lambda task_id, block, l_idx: f"task_{task_id}_block_{block}_l{l_idx}"
    layers_idxs = np.arange(len(model.h_fcs) + 1)

    for hl_idx in layers_idxs:
        id = get_act_id(model_at_task, block, hl_idx)
        # print("hook: ", id)
        if hl_idx == len(layers_idxs) - 1:
            if model.use_task_ro:
                model.fc_outs[model_at_task].register_forward_hook(get_activation(id))
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
            x = x[:, input_permutations[target_task]]
            _ = model(x, task_id=target_task)
        # store activations
        for hl_idx in layers_idxs:
            id = get_act_id(model_at_task, block, hl_idx)
            activations_from_layers.append(
                {
                    "label": label,
                    "task_id": model_at_task,
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


def collect_activations_for_models(
    mtypes, target_task, sub_loaders, input_permutations, eval_loader=None, **loaded_models_kwargs
):
    rows = []  # rows for DF
    print("loodind models and extracting activations: ")
    models_loaded = {}
    for m_type in mtypes:
        for t in SAVE_ON_TASKS:
            model_loaded = load_model(m_type, t, DEVICE, **loaded_models_kwargs)
            act_loaded = get_activations_for_model(
                model_loaded,
                model_at_task=t,
                block=0,
                target_task=target_task,
                input_permutations=input_permutations,
                sub_loaders=sub_loaders,
                device=DEVICE,
            )

            for entry in act_loaded:
                rows.append(
                    {
                        "model_type": m_type,
                        "task_id": entry["task_id"],
                        "block": entry["block"],
                        "layer": entry["layer"],
                        "label": entry["label"],
                        "activations": entry["activations"].detach().cpu(),
                    }
                )

            models_loaded[(m_type, t)] = model_loaded
            if eval_loader is not None:
                acc = evaluate(
                    model_loaded,
                    eval_loader,
                    0,
                    input_permutations[0],
                    device=DEVICE,
                )
                print(f"Model type: {m_type}, Task {t}, Accuracy on test set task 0: {acc:.4f}")
                print("-" * 80)

    acts_df = pd.DataFrame(rows)
    return acts_df, models_loaded


# region RSA
def compute_rdm(activations):
    # center each unit
    X = activations - activations.mean(axis=0, keepdims=True)
    # correlation between samples (rows)
    C = np.corrcoef(X)  # shape [N, N]
    RDM = 1.0 - C
    return RDM


def rdm_correlation(rdm1, rdm2):
    assert rdm1.shape == rdm2.shape
    idx = np.triu_indices_from(rdm1, k=1)
    v1 = rdm1[idx]
    v2 = rdm2[idx]

    v1_mean = v1.mean()
    v2_mean = v2.mean()
    num = np.sum((v1 - v1_mean) * (v2 - v2_mean))
    den = np.sqrt(np.sum((v1 - v1_mean) ** 2) * np.sum((v2 - v2_mean) ** 2))
    if den == 0:
        return np.nan
    return num / den


def rsa_between_models(acts_df, m1, m2, at_task, layers_idxs):
    scores = []
    for layer_idx in layers_idxs:
        a1 = acts_df.query("model_type == @m1 and layer == @layer_idx and task_id == @at_task")
        a2 = acts_df.query("model_type == @m2 and layer == @layer_idx and task_id == @at_task")
        # sort
        a1 = a1.sort_values(by=["block"])
        a2 = a2.sort_values(by=["block"])

        arctivations1 = a1["activations"].values
        arctivations2 = a2["activations"].values
        if len(arctivations1) == 0 or len(arctivations2) == 0:
            a1 = None
            a2 = None
        else:
            a1 = torch.cat(arctivations1.tolist(), dim=0).cpu().numpy()
            a2 = torch.cat(arctivations2.tolist(), dim=0).cpu().numpy()

        if a1 is None or a2 is None:
            scores.append(np.nan)
            continue
        rdm1 = compute_rdm(a1)
        rdm2 = compute_rdm(a2)
        score = rdm_correlation(rdm1, rdm2)
        scores.append(score)

    return scores


# region acc matrix
def cross_context_accuracy_matrix(
    model: MLP_SP,
    test_loader,
    input_permutations: np.ndarray,
    device,
    eval_task_ids=None,
    context_ids=None,
    **kwargs,
) -> np.ndarray:
    """
    Returns A with shape [len(eval_task_ids), len(context_ids)] where:
      A[i, j] = accuracy on eval_task_ids[i] when using context context_ids[j].
    """
    conserve_ro = model.use_task_ro if hasattr(model, "use_task_ro") else False
    if eval_task_ids is None:
        eval_task_ids = list(range(model.n_tasks))
    if context_ids is None:
        context_ids = list(range(model.n_tasks))

    acc_matrix = np.zeros((len(eval_task_ids), len(context_ids)), dtype=np.float32)

    for i, t in enumerate(eval_task_ids):
        for j, k in enumerate(context_ids):
            acc_matrix[i, j] = evaluate(
                model=model,
                test_loader=test_loader,
                task_id=k,
                perm=input_permutations[t],
                ro_id=t if conserve_ro else None,
                device=device,
                **kwargs,
            )
    return acc_matrix


# NOTE very simple context selectivity scores
def context_selectivity_scores(A: np.ndarray) -> dict:
    """
    Diagnostics from an accuracy matrix A (tasks x contexts):
      - diag: accuracy with correct context
      - offdiag_mean: mean accuracy with wrong contexts
      - selectivity: diag - offdiag_mean (higher => sharper keying)
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "Expect square task×context matrix."
    T = A.shape[0]
    diag = np.diag(A)
    off = A.copy()
    off[np.eye(T, dtype=bool)] = np.nan
    offdiag_mean = np.nanmean(off, axis=1)
    selectivity = diag - offdiag_mean
    return selectivity


# region ablations
def build_context_ablation_specs(model: MLP_SP) -> dict[str, list[int]]:

    L = len(model.hl_dims) + 1  # total context layers
    layers = list(range(L))
    return {
        "baseline": [],  # ablate nothing
        "all": layers,
        "input_only": list(range(1, L)),  # disable hidden contexts
        "hidden_only": [0],  # disable input context
        "early_only": list(range(2, L)),  # keep 0 and 1, disable 2..end
        "late_only": list(range(0, L - 1)),  # keep last only
    }


def layer_context_ablation_sweep(
    model: MLP_SP,
    test_loader,
    input_permutations: np.ndarray,
    device,
    ablation_specs: dict[str, list[int]] | None = None,
    eval_task_ids: list[int] | None = None,
    max_batches: int | None = None,
) -> pd.DataFrame:
    if eval_task_ids is None:
        eval_task_ids = list(range(model.n_tasks))

    if ablation_specs is None:
        ablation_specs = build_context_ablation_specs(model)

    # baseline (none)
    context_backups = {}
    rows = []

    for cond, ablate_layers in ablation_specs.items():
        ablate_complement_backup = set(ablate_layers) - set(context_backups.keys())
        backup_complement_ablate = set(context_backups.keys()) - set(ablate_layers)

        print(f"Ablation condition: {cond}, ablating layers: {ablate_layers}")
        # print(f"  - need to backup and ablate: {ablate_complement_backup}")
        # print(f"  - need to restore from backup: {backup_complement_ablate}")
        # print("context backups before ablation: ", context_backups.keys())
        for l in ablate_complement_backup:
            context_backups.update({l: model.__getattr__(f"context_{l}").clone()})
            setattr(model, f"context_{l}", torch.ones_like(getattr(model, f"context_{l}")))
        for m in backup_complement_ablate:
            setattr(model, f"context_{m}", context_backups[m])
            context_backups.pop(m)
        # print("context backups after ablation: ", context_backups.keys())

        accs = [
            evaluate(
                model,
                test_loader,
                task_id=t,
                perm=input_permutations[t],
                device=device,
                max_batches=max_batches,
            )
            for t in eval_task_ids
        ]

        for i, t in enumerate(eval_task_ids):
            rows.append(
                {
                    "condition": cond,
                    "task_id": int(t),
                    "acc": float(accs[i]),
                }
            )

    return pd.DataFrame(rows)


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
    ax.set_title(f"{title}\nLayer {layer_to_analyze}")
    ax.grid(True)

    if show_plot:
        plt.tight_layout()
        plt.draw()

    return ax


def plot_training_loss(loss, model_name, title="", ax=None, show_plot=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(loss, label=f"{model_name}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title(title)

    if show_plot:
        plt.tight_layout()
        plt.draw()

    return ax


def plot_task_accuracies(acc, model_name, title="", ax=None, show_plot=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(acc, label=f"{model_name}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)

    if show_plot:
        plt.tight_layout()
        plt.draw()

    return ax


def plot_cross_context_matrix(
    A: np.ndarray,
    task_ids=None,
    context_ids=None,
    vmin=0.0,
    vmax=1.0,
    title: str = "",
    ax=None,
    show_plot=False,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))

    im = ax.imshow(A, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Context used (k)")
    ax.set_ylabel("Eval task/data (t)")

    if task_ids is not None:
        ax.set_yticks(np.arange(len(task_ids)))
        ax.set_yticklabels([str(t) for t in task_ids])
    if context_ids is not None:
        ax.set_xticks(np.arange(len(context_ids)))
        ax.set_xticklabels([str(k) for k in context_ids], rotation=90)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if show_plot:
        plt.tight_layout()
        plt.draw()
    return ax


def plot_ablation_summary(df: pd.DataFrame, title: str = "", ax=None, show_plot=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    # line plot for each task
    for task_id, group in df.groupby("task_id"):
        ax.plot(
            group["condition"],
            group["acc"],
            marker="o",
            label=f"Task {task_id}",
        )
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Tasks", bbox_to_anchor=(1.05, 1), loc="upper left")

    if show_plot:
        plt.tight_layout()
        plt.draw()
    return ax


def plot_loo_heatmap(
    loo_df: pd.DataFrame,
    title: str = "",
    vmin=None,
    vmax=None,
    ax=None,
    show_plot=False,
):
    pivot = loo_df.pivot(index="task_id", columns="layer", values="delta_baseline").sort_index()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Context layer (ablated)")
    ax.set_ylabel("Task ID (eval)")

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns.tolist()])
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([str(r) for r in pivot.index.tolist()])

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ acc (full - ablated)")

    if show_plot:
        plt.tight_layout()
        plt.draw()
    return ax


def plot_loi_ranked_bars(
    loi_df: pd.DataFrame,
    title: str = "",
    ax=None,
    show_plot: bool = False,
):
    """
    LOI: keep only one context layer l active (others ablated).
    Visualize delta_all_off per layer with:
      - bars per task in different colors (grouped)
      - an additional "mean" bar per layer
    X-axis is ordered by layer index: 0..(n-1)
    """
    # expects columns: ["layer", "task_id", "delta_all_off"] (plus others ok)
    df = loi_df.copy()

    # enforce ordering by layer index (0..n-1)
    layers = sorted(df["layer"].unique().tolist())
    tasks = sorted(df["task_id"].unique().tolist())

    # pivot: rows=layer, cols=task_id
    piv = df.pivot_table(index="layer", columns="task_id", values="delta_all_off", aggfunc="mean").reindex(layers)
    layer_means = piv.mean(axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))

    x = np.arange(len(layers))
    n_tasks = len(tasks)
    group_width = 0.90  # total width allocated per layer
    n_bars = n_tasks + 1  # +1 for mean bar
    bar_w = group_width / n_bars

    # per-task bars (different colors)
    for i, t in enumerate(tasks):
        vals = piv[t].values if t in piv.columns else np.full(len(layers), np.nan)
        ax.bar(
            x - group_width / 2 + i * bar_w + bar_w / 2,
            vals,
            width=bar_w,
            label=f"Task {t}",
        )

    # mean bar (additional bar at the end of each group)
    ax.bar(
        x - group_width / 2 + n_tasks * bar_w + bar_w / 2,
        layer_means.values,
        width=bar_w,
        label="Mean",
        hatch="///",
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_ylabel("Δ acc (condition - all-off)")
    ax.set_xlabel("Context layer (only layer kept)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2, fontsize="small", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

    if show_plot:
        plt.tight_layout()
        plt.draw()
    return ax
