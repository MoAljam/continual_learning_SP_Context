import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# MNIST loader
def init_dataloader_mnist(batch_size, shuffle=True, **kwargs):

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  # val [0, 1]  # normalize to [-1, 1]
    )

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    def preprocess_data(dataset):  # just placeholder in case costume preprocessing steps is needed
        return dataset

    loader_train = DataLoader(preprocess_data(train_dataset), batch_size=batch_size, shuffle=shuffle, **kwargs)
    loader_test = DataLoader(preprocess_data(test_dataset), batch_size=batch_size, shuffle=False, **kwargs)

    return loader_train, loader_test


# get subset of dataloader with n samples of specified labels
def get_subset_of_labels_loader(dataloader, allowed_labels, n_samples_per_label):
    subset_images = []
    subset_labels = []
    label_counts = {label: 0 for label in allowed_labels}

    for images, labels in dataloader:
        for i in range(images.size(0)):
            label = labels[i].item()
            if label in allowed_labels and label_counts[label] < n_samples_per_label:
                subset_images.append(images[i])
                subset_labels.append(labels[i])
                label_counts[label] += 1

            if all(count >= n_samples_per_label for count in label_counts.values()):
                break
        if all(count >= n_samples_per_label for count in label_counts.values()):
            break

    subset_images = torch.stack(subset_images)
    subset_labels = torch.stack(subset_labels)

    subset_loader = DataLoader(
        TensorDataset(subset_images, subset_labels),
        batch_size=dataloader.batch_size,
        shuffle=False,  # important to keep order for analysis
    )
    return subset_loader


def train_one_epoch(model, optimizer, criterion, train_loader, task_id, perm, device):
    model.train()

    running_loss = 0.0
    for images, labels in train_loader:
        x = images.to(device)
        y = labels.to(device)
        B = images.size(0)

        # apply permutation
        x = x.view(B, -1)
        x = x[:, perm]

        logits = model(x, task_id=task_id)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * B
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


@torch.no_grad()
def evaluate(model, test_loader, task_id, perm, device, ro_id=None, max_batches=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for b_idx, (images, labels) in enumerate(test_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            x = images.to(device)
            y = labels.to(device)
            B = images.size(0)

            # apply permutation
            x = x.view(B, -1)
            x = x[:, perm]

            logits = model(x, task_id=task_id, ro_id=ro_id)
            _, predicted = torch.max(logits.data, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total
    return accuracy


def run_experiment(
    model,
    optimizer,
    criterion,
    train_loader,
    test_loader,
    input_permutations,
    save_on_tasks: list[int],
    blocks: int,
    n_tasks: int,
    epochs_per_task: int,
    device,
    save_dir="./models_checkpoints",
    eval_on_tasks: list[int] = [0],
):

    # NOTE not good but quick lazy way
    trace = {
        "loss": [],
        "training_on_task": [],
        "accuracy_task_curr": [],  # very redundant
        **{f"accuracy_task_{t_id}": [] for t_id in eval_on_tasks},
    }

    model_type = "c" if model.use_context else "b"
    model_type += "_tro" if model.use_task_ro else ""
    os.makedirs(save_dir, exist_ok=True)

    c_m_mask = "_".join([str(n) for n in model.context_layers_mask])

    total_epochs = epochs_per_task * n_tasks * blocks
    pbar = tqdm(
        range(total_epochs),
        desc=f"g_epochs:",
        leave=True,
        total=total_epochs,
    )
    for block in range(blocks):
        # TODO randomize task order, keep track of the corresponding permutation vectors
        for task_id in range(n_tasks):
            # reset momentum of optimizer
            optimizer.state.clear()

            for epoch in range(epochs_per_task):
                epoch_loss = train_one_epoch(
                    model, optimizer, criterion, train_loader, task_id, input_permutations[task_id], device
                )

                avg_loss = epoch_loss / len(train_loader)
                trace["loss"].append(avg_loss)
                trace["training_on_task"].append(task_id)
                pbar.set_postfix(
                    Loss=f"{avg_loss:.4f}",
                    B=f"{block+1}/{blocks}",
                    T=f"{task_id+1}/{n_tasks}",
                    E=f"{epoch+1}/{epochs_per_task}",
                )
                pbar.update(1)

            if task_id in save_on_tasks:
                # save model (checkpoint)
                model_path = os.path.join(
                    save_dir, f"model_{model_type}_task_{task_id}_block_{block}_cm_{c_m_mask}.pth"
                )
                torch.save(model.state_dict(), model_path)
                tqdm.write(f"(checkpoint) model saved at {model_path}")

            # eval on both original and current task
            # NOTE redundant
            accuracy_curr_task = evaluate(
                model, test_loader, task_id=task_id, perm=input_permutations[task_id], device=device
            )
            trace["accuracy_task_curr"].append(accuracy_curr_task)
            tqdm.write(f"current task ({task_id}) acc: {accuracy_curr_task*100:.2f}% (Block {block})")

            for t_id in eval_on_tasks:
                accuracy = evaluate(model, test_loader, task_id=t_id, perm=input_permutations[t_id], device=device)
                trace[f"accuracy_task_{t_id}"].append(accuracy)
                tqdm.write(f"Task {t_id} acc: {accuracy*100:.2f}% (Block {block}, Task {task_id})")

    pbar.close()

    return trace
