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

from analysis import *

# lazy import of globals, model-explicit meta-parameters are saved with training traces in checkpoints
from main import (
    DEVICE,
    BLOCKS,
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

PLOTS_DIR = "./plots"


def adjust_ax_to_tasks(ax, block, n_tasks, epochs_per_task, max_ticks=20):
    # xticks for tasks on an additional x-axis
    total_epochs = block * n_tasks * epochs_per_task
    step = max(1, total_epochs // max_ticks)
    task_ticks = np.arange(0, total_epochs + 1, step)
    task_ticks = task_ticks
    task_labels = [str(x % n_tasks) for x in task_ticks]

    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xticks(task_ticks)
    ax_top.set_xticklabels(task_labels)
    ax_top.set_xlabel("Task ID")

    # vertical lines at block and task boundaries
    task_lines = np.arange(0, block * n_tasks * epochs_per_task + 1, epochs_per_task)[1:-1]
    block_lines = np.arange(0, block * n_tasks * epochs_per_task + 1, n_tasks * epochs_per_task)[1:-1]
    for epoch in task_lines:
        ax.axvline(x=epoch, color="gray", linestyle="--", linewidth=1, alpha=0.2)
    for epoch in block_lines:
        ax.axvline(x=epoch, color="black", linestyle="--", linewidth=2, alpha=0.4)

    return ax


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
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
    print("### sub loaders contain:", {key: len(sub_loaders[key].dataset) for key in sub_loaders})

    # load input permutations
    input_permutations = np.load("mnist_permutations.npy")

    # region accuracy & loss
    print("### accuracy and loss on saved traces")
    cm = "_".join([str(n) for n in CONTEXT_LAYERS_MASK])
    saved_traces_available = False
    try:
        with open(os.path.join(CHECKPOINT_DIR, f"trace_b_cm_{cm}.json"), "r") as f:
            trace_baseline = json.load(f)
        with open(os.path.join(CHECKPOINT_DIR, f"trace_c_cm_{cm}.json"), "r") as f:
            trace_contex = json.load(f)
        with open(os.path.join(CHECKPOINT_DIR, f"trace_b_tro_cm_{cm}.json"), "r") as f:
            trace_baseline_tro = json.load(f)
        with open(os.path.join(CHECKPOINT_DIR, f"trace_c_tro_cm_{cm}.json"), "r") as f:
            trace_contex_tro = json.load(f)
        saved_traces_available = True
        print("Loaded saved traces")
    except FileNotFoundError as e:
        print("Saved traces not found. re-run training to generate traces.")
        print("error:\n", e)

    if not saved_traces_available:
        print("no available traces, skipping plotting trained traces")
    else:
        print("Plotting saved traces")
        # plot results: accuracy on task 0
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs = axs
        plot_task_accuracies(trace_contex["accuracy_task_0"], "Context Model", ax=axs[0])
        plot_task_accuracies(trace_baseline["accuracy_task_0"], "Baseline Model", ax=axs[0])

        plot_task_accuracies(trace_contex_tro["accuracy_task_0"], "Context Model TRO", ax=axs[1])
        plot_task_accuracies(trace_baseline_tro["accuracy_task_0"], "Baseline Model TRO", ax=axs[1])

        axs[0].set_title("CL: (Task 0) same readout")
        axs[1].set_title("CL: (Task 0) task-specific readout")

        axs[0].legend()
        axs[1].legend()

        for ax in axs:
            adjust_ax_to_tasks(ax, BLOCKS, N_TASKS, EPOCHS_PER_TASK)

        plt.tight_layout()
        if PLOTS_DIR:
            plt.savefig(os.path.join(PLOTS_DIR, "cl_performance_task_0.png"))
        plt.show(block=False)

        # plot results: loss and accuracy on current task to ensure the models are learning properly on the new tasks
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs = axs.flatten()

        plot_task_accuracies(trace_contex["accuracy_task_curr"], "Context Model", ax=axs[0])
        plot_task_accuracies(trace_baseline_tro["accuracy_task_curr"], "Baseline Model TRO", ax=axs[0])
        plot_task_accuracies(trace_contex_tro["accuracy_task_curr"], "Context Model TRO", ax=axs[0])
        plot_task_accuracies(trace_baseline["accuracy_task_curr"], "Baseline Model", ax=axs[0])

        plot_training_loss(trace_contex["loss"], "Context Model", title="Training Loss", ax=axs[1])
        plot_training_loss(trace_contex_tro["loss"], "Context Model TRO", title="Training Loss", ax=axs[1])
        plot_training_loss(trace_baseline["loss"], "Baseline Model", title="Training Loss", ax=axs[1])
        plot_training_loss(trace_baseline_tro["loss"], "Baseline Model TRO", title="Training Loss", ax=axs[1])

        axs[0].set_title("CL: (current task)")
        axs[1].set_title("Training loss")
        # add vertical lines at block and task boundaries
        for ax in axs:
            adjust_ax_to_tasks(ax, BLOCKS, N_TASKS, EPOCHS_PER_TASK)

        h, l = axs[0].get_legend_handles_labels()
        fig.suptitle("", fontsize=16)
        fig.legend(h, l, loc="upper center", ncol=4, fontsize="small", frameon=False, bbox_to_anchor=(0.5, 0.99))
        plt.tight_layout()
        if PLOTS_DIR:
            plt.savefig(os.path.join(PLOTS_DIR, "cl_performance_task_curr.png"))
        plt.show(block=False)

    # region acc matrix
    print("### cross-context accuracy matrix")
    mtypes = ["c", "b", "c_tro", "b_tro"]
    at_tasks = [9]

    eval_task_ids = list(range(N_TASKS))
    context_ids = list(range(N_TASKS))
    # eval_task_ids = [0, 4, 8, 9]
    # context_ids = [0, 4, 8, 9]

    results = {}
    for m_type in mtypes:
        for ckpt_t in at_tasks:
            model = load_model(m_type, ckpt_t, DEVICE)

            acc_matrix = cross_context_accuracy_matrix(
                model=model,
                test_loader=test_loader,
                input_permutations=input_permutations,
                device=DEVICE,
                eval_task_ids=eval_task_ids,
                context_ids=context_ids,
            )
            results[(m_type, ckpt_t)] = {
                "acc_matrix": acc_matrix,
                "score": context_selectivity_scores(acc_matrix) if (len(eval_task_ids) == len(context_ids)) else None,
            }

    for (m_type, ckpt_t), res in results.items():
        acc_matrix = res["acc_matrix"]
        scores = res["score"]
        print(f"Model: {m_type}")
        print("context selectivity scores: ", scores, "\nmean selectivity: ", np.mean(scores))

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        ax = axs[0]

        fig.suptitle(f"model: {m_type}", fontsize=16)
        plot_cross_context_matrix(
            acc_matrix,
            eval_task_ids,
            context_ids,
            title=f"Cross-Context Accuracy",
            ax=ax,
        )
        ax = axs[1]
        # accuracy of the correct matching task and mean of with unmatching tasks to context
        diag = np.diag(acc_matrix)
        offdiag = acc_matrix.copy()
        offdiag[np.eye(len(offdiag), dtype=bool)] = np.nan
        offdiag_mean = np.nanmean(offdiag, axis=0)
        x = np.arange(len(diag))
        ax.plot(x, diag, marker="o", label="correct context")
        ax.plot(x, offdiag_mean, marker="o", label="mean wrong contexts")
        ax.set_xlabel("Task ID")
        ax.set_ylabel("Accuracy")
        ax.set_title("Correct Context vs Mean Wrong Contexts")
        ax.set_xticks(x)
        ax.set_xticklabels([str(tid) for tid in eval_task_ids])
        ax.legend()

        plt.tight_layout()
        if PLOTS_DIR:
            plt.savefig(os.path.join(PLOTS_DIR, f"acc_matrix_{m_type}_task_{ckpt_t}.png"))
        plt.show(block=False)

    # region rep. shift
    # get activations of one model of multiple tasks
    print("### representational shift analysis")
    target_task = 0
    mtypes = ["c", "b", "c_tro", "b_tro"]

    labels = [4, 7]
    num_samples_per_label = 15
    sub_loaders = {
        labels[i]: get_subset_of_labels_loader(
            train_loader, allowed_labels=[labels[i]], n_samples_per_label=num_samples_per_label
        )
        for i in range(len(labels))
    }

    acts_df, loaded_models = collect_activations_for_models(
        mtypes,
        target_task,
        sub_loaders,
        input_permutations,
        # eval_loader=test_loader,
        eval_loader=None,
        context_layers_mask=CONTEXT_LAYERS_MASK,
    )
    model_loaded = loaded_models[("c", SAVE_ON_TASKS[-1])]
    print("df shape:", acts_df.shape)
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
        fig.suptitle(f"representational shift over tasks", fontsize=16)
        fig.legend(
            h,
            l,
            loc="upper center",
            ncol=3,
            fontsize="small",
            frameon=False,
            bbox_to_anchor=(0.5, 0.95),
        )

        plt.tight_layout()
        if PLOTS_DIR:
            plt.savefig(os.path.join(PLOTS_DIR, f"representational_shift_layer_{layer_to_analyze}.png"))
        plt.show(block=False)

    # region ablation
    print("### context ablation analysis")
    mtypes = ["c", "c_tro"]
    at_tasks = [9]
    eval_task_ids = [0, 3, 6, 9]
    num_context_layers = len(CONTEXT_LAYERS_MASK)
    ablation_specs = {
        "baseline": [],  # ablate nothing
        "all": list(range(num_context_layers)),
        # LOO
        **{f"LOO_l_{i}": [i] for i in range(num_context_layers)},
        # LOI
        **{f"LOI_l_{i}": [j for j in range(num_context_layers) if j != i] for i in range(num_context_layers)},
    }

    results_dfs = []
    for m_type in mtypes:
        for ckpt_t in at_tasks:
            model = load_model(m_type, ckpt_t, DEVICE, CONTEXT_LAYERS_MASK)

            ablation_results = layer_context_ablation_sweep(
                model=model,
                test_loader=test_loader,
                input_permutations=input_permutations,
                device=DEVICE,
                eval_task_ids=eval_task_ids,
                ablation_specs=ablation_specs,
            )
            ablation_results["model_type"] = m_type
            results_dfs.append(ablation_results)
    results_all_df = pd.concat(results_dfs, ignore_index=True)

    baseline_ref = results_all_df.query("condition == 'baseline'").set_index(["model_type", "task_id"])["acc"].to_dict()
    alloff_ref = results_all_df.query("condition == 'all'").set_index(["model_type", "task_id"])["acc"].to_dict()
    results_all_df["delta_baseline"] = np.nan
    results_all_df["delta_all_off"] = np.nan
    for idx, row in results_all_df.iterrows():
        key = (row["model_type"], row["task_id"])
        b = baseline_ref.get(key, np.nan)  # baseline = no ablation
        a = alloff_ref.get(key, np.nan)  # all-off = all contexts ablated
        # baseline - condition (drop relative to intact context)
        results_all_df.at[idx, "delta_baseline"] = float(b - row["acc"]) if np.isfinite(b) else np.nan
        # condition - all-off (gain relative to no context anywhere)
        results_all_df.at[idx, "delta_all_off"] = float(row["acc"] - a) if np.isfinite(a) else np.nan

    print("Ablation results df shape:", results_all_df.shape)  # (72, 5)
    print(results_all_df.columns)

    # columns: condition, task_id, acc, model_type
    # plot results
    for m_type in mtypes:
        df_m = results_all_df.query("model_type == @m_type").copy()
        df_loo = df_m.loc[results_all_df["condition"].str.startswith("LOO_l_")].copy()
        df_loo.loc[:, "layer"] = df_loo["condition"].str.split("_").str[-1].astype(int)
        df_loi = df_m.loc[results_all_df["condition"].str.startswith("LOI_l_")].copy()
        df_loi.loc[:, "layer"] = df_loi["condition"].str.split("_").str[-1].astype(int)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        plot_ablation_summary(
            df_m,
            title=f"Context Ablation Analysis | Model: {m_type}",
            ax=ax,
        )
        plt.tight_layout()
        if PLOTS_DIR:
            plt.savefig(os.path.join(PLOTS_DIR, f"context_ablation_{m_type}.png"))
        plt.show(block=False)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        ax = axs[0]
        plot_loo_heatmap(
            df_loo,
            title=f"LOO Heatmap | Model: {m_type}",
            ax=ax,
            show_plot=True,
        )
        ax = axs[1]
        plot_loi_ranked_bars(
            df_loi,
            title=f"LOI Ranked Bar Chart | Model: {m_type}",
            ax=ax,
            show_plot=True,
        )
        plt.tight_layout()
        if PLOTS_DIR:
            plt.savefig(os.path.join(PLOTS_DIR, f"context_ablation_loo_loi_{m_type}.png"))
        plt.show(block=False)

    # region RSA c vs b
    # RSA analysis between context and baseline models at a specific task
    print("### RSA analysis between context and baseline models")
    mtypes = ["c", "b", "c_tro", "b_tro"]
    target_tasks = [0, 5, 9]
    model_at_task = 9
    labels = np.arange(OUT_DIM)
    num_samples_per_label = 10

    sub_loaders = {
        labels[i]: get_subset_of_labels_loader(
            train_loader, allowed_labels=[labels[i]], n_samples_per_label=num_samples_per_label
        )
        for i in range(len(labels))
    }
    print("sub loaders contain:", {key: len(sub_loaders[key].dataset) for key in sub_loaders})
    rdms = {}
    rsa_scores = {}
    for target_task in target_tasks:
        print(f"### RSA analysis for task {target_task}")
        acts_df, loaded_models = collect_activations_for_models(
            mtypes,
            target_task,
            sub_loaders,
            input_permutations,
            # eval_loader=test_loader,
            eval_loader=None,
        )
        layers_idxs = np.arange(len(model_loaded.h_fcs) + 1)
        rdms[target_task] = {}
        rsa_scores[target_task] = []
        for layer_idx in layers_idxs:
            acts_c = acts_df.query("model_type == 'c' and layer == @layer_idx and task_id == @model_at_task").copy()
            acts_b = acts_df.query("model_type == 'b' and layer == @layer_idx and task_id == @model_at_task").copy()
            # sort
            acts_c = acts_c.sort_values(by=["block"])
            acts_b = acts_b.sort_values(by=["block"])

            acts_c = torch.cat(acts_c["activations"].tolist(), dim=0).cpu().numpy()
            acts_b = torch.cat(acts_b["activations"].tolist(), dim=0).cpu().numpy()

            if acts_c is None or acts_b is None:
                rsa_scores[target_task].append(np.nan)
                continue

            rdm_c = compute_rdm(acts_c)
            rdm_b = compute_rdm(acts_b)
            rdms[target_task][layer_idx] = {"context": rdm_c, "baseline": rdm_b}

            corr = rdm_correlation(rdm_c, rdm_b)
            rsa_scores[target_task].append(corr)

        # plot rdms of all layers
        num_layers = len(layers_idxs)
        fig, axs = plt.subplots(2, num_layers, figsize=(5 * num_layers, 10))
        fig.suptitle(f"RDMs of Context and Baseline Models | Task {target_task}", fontsize=16)

        for layer_idx in layers_idxs:
            rdm_c = rdms[target_task][layer_idx]["context"]
            rdm_b = rdms[target_task][layer_idx]["baseline"]

            ax_1, ax_2 = axs[1, layer_idx], axs[0, layer_idx]
            im1 = ax_1.imshow(rdm_c, cmap="viridis")
            ax_1.set_title(f"Layer {layer_idx} - Context Model")
            plt.colorbar(im1, ax=ax_1, fraction=0.046, pad=0.04)
            im2 = ax_2.imshow(rdm_b, cmap="viridis")
            ax_2.set_title(f"Layer {layer_idx} - Baseline Model")
            plt.colorbar(im2, ax=ax_2, fraction=0.046, pad=0.04)

            ticks_positions = (
                np.arange(0, len(labels) * num_samples_per_label, num_samples_per_label)
                + (num_samples_per_label - 1) / 2
            )
            ticks_labels = [str(label) for label in labels]
            # ticks for highlighting same labels
            for ax in [ax_1, ax_2]:
                ax.set_xticks(ticks_positions)
                ax.set_xticklabels(ticks_labels)
                ax.set_yticks(ticks_positions)
                ax.set_yticklabels(ticks_labels)
                ax.tick_params(axis="x", rotation=90)

        plt.tight_layout()
        if PLOTS_DIR:
            plt.savefig(os.path.join(PLOTS_DIR, f"rdms_task_{target_task}.png"))
        plt.show(block=False)

    # plot the RSA similarity across layers
    plt.figure(figsize=(6, 4))
    for target_task in target_tasks:
        rsa_scores_cb = rsa_scores[target_task]
        layers_idxs = np.arange(len(model_loaded.h_fcs) + 1)
        plt.plot(layers_idxs, rsa_scores_cb, marker="o", label=f"Task {target_task}")
    plt.xlabel("Layer index")
    plt.ylabel("RDM correlation (Context vs Baseline)")
    plt.title(f"RSA: Context vs Baseline across layers")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if PLOTS_DIR:
        plt.savefig(os.path.join(PLOTS_DIR, f"rsa_context_vs_baseline_task_{target_task}.png"))
    plt.show(block=False)
