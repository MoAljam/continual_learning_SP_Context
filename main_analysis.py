import os

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
from Cheung_2019 import (
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
)


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


if __name__ == "__main__":
    # load MNIST
    train_loader, test_loader = init_dataloader_mnist(batch_size=128, shuffle=True)

    labels = [4, 7]
    num_samples_per_label = 15

    sub_loaders = {
        labels[i]: get_subset_of_labels_loader(
            train_loader, allowed_labels=[labels[i]], n_samples_per_label=num_samples_per_label
        )
        for i in range(len(labels))
    }
    print("sub loaders contain:", {key: len(sub_loaders[key].dataset) for key in sub_loaders})

    # load input permutations
    input_permutations = np.load("mnist_permutations.npy")
    target_task = 0

    # get activations of one model of multiple tasks
    mtypes = ["c", "b", "c_tro", "b_tro"]

    rows = []  # rows for DF
    print("loodind models and extracting activations: ")
    for m_type in mtypes:
        use_context = True if (m_type == "c" or m_type == "c_tro") else False
        use_task_ro = True if (m_type == "c_tro" or m_type == "b_tro") else False

        model_loaded = MLP_SP(
            input_dim=IN_DIM,
            output_dim=OUT_DIM,
            hl_dims=HL_DIMS,
            num_tasks=N_TASKS,
            use_context=use_context,
            use_task_ro=use_task_ro,
            device=DEVICE,
            context_layers_mask=CONTEXT_LAYERS_MASK,
        )

        for t in SAVE_ON_TASKS:
            path = os.path.join(CHECKPOINT_DIR, f"model_{m_type}_task_{t}_block_0.pth")
            state = torch.load(path, map_location=DEVICE)
            model_loaded.load_state_dict(state)
            model_loaded.to(DEVICE)

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

            acc = evaluate(
                model_loaded,
                test_loader,
                0,
                input_permutations[0],
                device=DEVICE,
            )
            print(f"Model type: {m_type}, Task {t}, Accuracy on test set task 0: {acc:.4f}")
        print("-" * 50)

    acts_df = pd.DataFrame(rows)

    print("DataFrame shape:", acts_df.shape)
    print(acts_df.columns)

    # get PCA of activations from models of a specific layer and tasks, using acts_df
    layers_to_analyze = [0, 1, 2]
    at_tasks = [0, 5, 9]

    for layer_to_analyze in layers_to_analyze:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.flatten()

        for idx, m_type in enumerate(mtypes):
            # select relevant rows from the DataFrame
            df_m = acts_df.query("model_type == @m_type and layer == @layer_to_analyze and task_id in @at_tasks").copy()

            # (optional but nice) sort for deterministic ordering in PCA / plotting
            df_m = df_m.sort_values(by=["task_id", "block"])

            #
            act_relevant = [
                {
                    "label": row["label"],
                    "task_id": row["task_id"],
                    "block": row["block"],
                    "layer": row["layer"],
                    "activations": row["activations"],
                }
                for _, row in df_m.iterrows()
            ]

            # concat into single tensor for PCA
            activations_array = torch.cat(df_m["activations"].tolist(), dim=0).cpu().numpy()

            # get PCA
            activations_pca = get_activations_pca(activations_array, n_pcs=2)

            # plot PCA (same function as before)
            plot_activations_pca(
                act_relevant,
                activations_pca,
                layer_to_analyze,
                ax=axs[idx],
                title=f"Model Type: {m_type}",
            )

        h, l = axs[0].get_legend_handles_labels()
        fig.legend(
            h,
            l,
            loc="upper center",
            ncol=3,
            fontsize="small",
            frameon=False,
            bbox_to_anchor=(0.5, 1.05),
        )

        plt.tight_layout()
        plt.show()

    # RSA analysis between context and baseline models at a specific task
    mtypes = ["c", "b", "c_tro", "b_tro"]

    at_task = 5
    layers_idxs = np.arange(len(model_loaded.h_fcs) + 1)

    rsa_scores_cb = []
    rdms = {}
    for layer_idx in layers_idxs:
        acts_c = acts_df.query("model_type == 'c' and layer == @layer_idx and task_id == @at_task").copy()
        acts_b = acts_df.query("model_type == 'b' and layer == @layer_idx and task_id == @at_task").copy()
        # sort
        acts_c = acts_c.sort_values(by=["block"])
        acts_b = acts_b.sort_values(by=["block"])

        acts_c = torch.cat(acts_c["activations"].tolist(), dim=0).cpu().numpy()
        acts_b = torch.cat(acts_b["activations"].tolist(), dim=0).cpu().numpy()

        if acts_c is None or acts_b is None:
            rsa_scores_cb.append(np.nan)
            continue

        rdm_c = compute_rdm(acts_c)
        rdm_b = compute_rdm(acts_b)
        rdms[layer_idx] = {"context": rdm_c, "baseline": rdm_b}

        corr = rdm_correlation(rdm_c, rdm_b)
        rsa_scores_cb.append(corr)

    # plot rdms of all layers
    num_layers = len(layers_idxs)
    fig, axs = plt.subplots(2, num_layers, figsize=(5 * num_layers, 10))
    fig.suptitle(f"RDMs of Context and Baseline Models | Task {at_task}", fontsize=16)

    for layer_idx in layers_idxs:
        rdm_c = rdms[layer_idx]["context"]
        rdm_b = rdms[layer_idx]["baseline"]

        ax_1, ax_2 = axs[0, layer_idx], axs[1, layer_idx]
        im1 = ax_1.imshow(rdm_c, cmap="viridis")
        ax_1.set_title(f"Layer {layer_idx} - Context Model")
        plt.colorbar(im1, ax=ax_1, fraction=0.046, pad=0.04)
        im2 = ax_2.imshow(rdm_b, cmap="viridis")
        ax_2.set_title(f"Layer {layer_idx} - Baseline Model")
        plt.colorbar(im2, ax=ax_2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # plot the RSA similarity across layers
    plt.figure(figsize=(6, 4))
    plt.plot(layers_idxs, rsa_scores_cb, marker="o")
    plt.xlabel("Layer index")
    plt.ylabel("RDM correlation (Context vs Baseline)")
    plt.title(f"RSA: Context vs Baseline across layers | Task {at_task} ")
    plt.grid(True)
    plt.show()

    # with and without task-specific readout
    layers_idxs = np.arange(len(model_loaded.h_fcs) + 1)
    rsa_cb = rsa_between_models(acts_df, "c", "b", at_task, layers_idxs)
    rsa_cb_tro = rsa_between_models(acts_df, "c_tro", "b_tro", at_task, layers_idxs)

    plt.figure(figsize=(6, 4))
    plt.plot(layers_idxs, rsa_cb, marker="o", label="Context vs Baseline")
    plt.plot(layers_idxs, rsa_cb_tro, marker="s", label="Context TRO vs Baseline TRO")
    plt.xlabel("Layer index")
    plt.ylabel("RDM correlation")
    plt.title(f"RSA across layers | Task {at_task} ")
    plt.legend()
    plt.grid(True)
    plt.show()
