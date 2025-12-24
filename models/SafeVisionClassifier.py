import os
import json
import torch
import argparse
import warnings
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel
from model_Multitask3 import (
    AdvancedClassifierHead_CLIP,
    MLPHead,
    MultiHeadCLIPClassifier,
    Multi_MultiC_GramCluster_v2
)
from util import fix_seed
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from PIL import Image
import psutil
import random

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Task type
task_type_dict = {
    "object": "multiclass",
    "style": "multiclass",
    "nsfw": "multiclass",
}

# =======================
# Auto-estimate batch_size_save
# =======================
def estimate_batch_size(dataset_path, target_mem_ratio=0.8, img_size=(224, 224), channels=3, dtype_bytes=4):
    mem = psutil.virtual_memory()
    avail_bytes = mem.available
    safe_bytes = avail_bytes * target_mem_ratio
    img_bytes = img_size[0] * img_size[1] * channels * dtype_bytes
    batch_size = int(safe_bytes // img_bytes)
    batch_size = max(1, batch_size)
    print(f"[INFO] Available RAM: {avail_bytes/1e9:.2f} GB, estimated batch_size_save: {batch_size}")
    return batch_size

# =======================
# Save dataset in batches
# =======================
def save_dataset_to_pth(dataset_path, save_path, batch_size=512, img_size=(224, 224)):
    os.makedirs(save_path, exist_ok=True)
    if any(f.endswith(".pth") for f in os.listdir(save_path)):
        print(f"[INFO] Dataset already saved in {save_path}, skipping...")
        return

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    ds = datasets.ImageFolder(dataset_path)
    all_tensors, all_labels, all_domains = [], [], []
    batch_idx = 0

    print(f"[INFO] Loading dataset from {dataset_path} into memory in batches and saving to {save_path}")
    for i, (img_path, _) in enumerate(tqdm(ds.samples)):
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img)
        all_tensors.append(tensor)

        label = os.path.basename(os.path.dirname(img_path))
        all_labels.append(label)

        if "sd" in os.path.basename(img_path).lower():
            all_domains.append("sdgen")
        else:
            all_domains.append("real")

        if (i + 1) % batch_size == 0 or (i + 1) == len(ds):
            batch_file = os.path.join(save_path, f"batch{batch_idx}.pth")
            torch.save({
                "data": torch.stack(all_tensors),
                "labels": all_labels,
                "domains": all_domains
            }, batch_file)

            batch_idx += 1
            all_tensors, all_labels, all_domains = [], [], []

# =======================
# MemoryDataset
# =======================
class MemoryDataset(Dataset):
    def __init__(self, pth_dir, augment=False, task_name=None):
        self.pth_files = sorted([os.path.join(pth_dir, f) for f in os.listdir(pth_dir) if f.endswith(".pth")])
        self.augment = augment
        self.task_name = task_name
        self.data_dicts = []

        for f in self.pth_files:
            self.data_dicts.append(torch.load(f, map_location="cpu"))

        all_labels = []
        for d in self.data_dicts:
            all_labels.extend(d["labels"])
        self.label2idx = {l: i for i, l in enumerate(sorted(set(all_labels)))}

        if augment and task_name == "style":
            self.real_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(5),
                transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),
            ])
            self.gen_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        else:
            self.real_transform = None
            self.gen_transform = None

        self.flat_data = [(d_idx, i) for d_idx, d in enumerate(self.data_dicts) for i in range(len(d["labels"]))]

    def __len__(self):
        return len(self.flat_data)

    def __getitem__(self, idx):
        d_idx, i = self.flat_data[idx]
        data_dict = self.data_dicts[d_idx]
        image = data_dict["data"][i]
        label_str = data_dict["labels"][i]
        label = self.label2idx[label_str]
        domain = data_dict["domains"][i]

        if self.augment and self.task_name == "style":
            if domain == "real" and self.real_transform:
                image = self.real_transform(transforms.ToPILImage()(image))
                image = transforms.ToTensor()(image)
            elif domain == "sdgen" and self.gen_transform:
                image = self.gen_transform(transforms.ToPILImage()(image))
                image = transforms.ToTensor()(image)

        image = image * 2.0 - 1.0

        return {
            "pixel_values": image,
            "labels": torch.tensor(label, dtype=torch.long),
            "label_str": label_str,
        }

def make_loader_from_pth(pth_dir, batch_size, augment=False, task_name=None):
    dataset = MemoryDataset(pth_dir, augment=augment, task_name=task_name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=augment, num_workers=0)

# =======================
# load_clip
# =======================
def load_clip():
    clip_model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    return clip_model, processor

# =======================
# PGD adversarial attack
# =======================
def pgd_attack(model, images, labels, task, eps=8/255, alpha=2/255, iters=3):
    model.eval()
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    delta = torch.zeros_like(images).uniform_(-eps, eps).to(device)
    delta.requires_grad = True

    for _ in range(iters):
        outputs = model(images + delta, task_name=task, labels=labels, detach_clip=False)
        loss = outputs["loss"]
        loss.backward()
        grad = delta.grad.detach()
        delta.data = (delta + alpha * torch.sign(grad)).clamp(-eps, eps)
        delta.grad.zero_()
    adv_images = (images + delta).clamp(-1, 1)
    return adv_images

# =======================
# Compute attack success rate
# =======================
def compute_attack_success_rate(model, pixel_values, labels, task):
    with torch.no_grad():
        outputs = model(pixel_values, task_name=task, labels=labels, detach_clip=True)
        logits = outputs['logits']

    if task == 'nsfw':
        preds = torch.argmax(logits, dim=1)
        success = (preds != labels).float().sum().item()
    else:
        if task_type_dict[task] == 'multiclass':
            preds = torch.argmax(logits, dim=1)
            success = (preds != labels).float().sum().item()
        else:
            preds = (torch.sigmoid(logits.squeeze()) > 0.5).long()
            success = (preds != labels).float().sum().item()
    return success / labels.size(0)

# =======================
# Evaluation
# =======================
def evaluate(model, loader, task, adv=False, eps=8/255, alpha=2/255, iters=3):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    total_success, total_batches = 0, 0
    task_type = task_type_dict[task]

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        if adv:
            pixel_values = pgd_attack(model, pixel_values, labels, task, eps=eps, alpha=alpha, iters=iters)
            total_success += compute_attack_success_rate(model, pixel_values, labels, task)
            total_batches += 1

        with torch.no_grad():
            outputs = model(pixel_values, task_name=task, labels=labels, detach_clip=True)

        logits = outputs["logits"]

        if task == "nsfw":
            preds = torch.argmax(logits, dim=1)
        else:
            if task_type == "multiclass":
                preds = torch.argmax(logits, dim=1)
            else:
                preds = (torch.sigmoid(logits.squeeze()) > 0.5).long()
        total_correct += (preds == labels).sum().item()

        total_loss += outputs["loss"].item() * labels.size(0)
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_success = total_success / max(1, total_batches) if adv else None
    return avg_loss, avg_acc, avg_success

# =======================
# Plotting
# =======================
def plot_metrics(metrics, save_dir, task):
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(["loss", "acc", "attack_success"]):
        plt.subplot(1, 3, i + 1)
        for split in metrics:
            if metric in metrics[split]:
                plt.plot(metrics[split][metric], label=split, marker='o')
        plt.title(f"{task.capitalize()} {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.upper())
        plt.legend()
        plt.grid(True)
        if metric == "acc" or metric == "attack_success":
            plt.ylim(0.0, 1.05)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{task}_metrics.png"))
    plt.close()

# =======================
# Main
# =======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-dir", type=str, default="xxx")
    parser.add_argument("--results-dir", type=str, default="xxx")
    parser.add_argument("--batch-size-save", type=int, default=None)
    parser.add_argument("--adv-ratio", type=float, default=0.5)
    args = parser.parse_args()

    fix_seed(42)
    os.makedirs(args.results_dir, exist_ok=True)
    clip_model, processor = load_clip()

    # === Prepare data ===
    datasets_to_save = [
        ("object/train", "object_train"),
        ("object/val/Real", "object_val_real"),
        ("object/val/SD-Gen", "object_val_sdgen"),
        ("style/train", "style_train"),
        ("style/val/Real", "style_val_real"),
        ("style/val/SD-Gen", "style_val_sdgen"),
        ("nsfw/train", "nsfw_train"),
        ("nsfw/val/Real", "nsfw_val_real"),
        ("nsfw/val/SD-Gen", "nsfw_val_sdgen"),
    ]
   
    for src, dst in datasets_to_save:
        dataset_path = os.path.join(args.base_dir, src)
        save_path = os.path.join("xxx", dst)
        if args.batch_size_save is None:
            batch_size_save = estimate_batch_size(dataset_path)
        else:
            batch_size_save = args.batch_size_save
        save_dataset_to_pth(dataset_path, save_path, batch_size=batch_size_save)

    # === Build model ===
    head_dict = {
        "object": AdvancedClassifierHead_CLIP(input_dim=512, hidden_dim=512, num_classes=10),
        "style": Multi_MultiC_GramCluster_v2(
            input_dim=512, hidden_dim=512, num_classes=20,
            dropout_rate=0.3, gram_reduce_dim=512, cluster_factor=3
        ),
        "nsfw": MLPHead(input_dim=512, hidden_dim=256, num_classes=7, dropout=0.3),
    }
    model = MultiHeadCLIPClassifier(clip_model, head_dict, task_type_dict).to(device)

    for task in ["object", "style", "nsfw"]:
        log_file_path = os.path.join(args.results_dir, f"{task}_epoch_logs.json")
        with open(log_file_path, "w") as f:
            json.dump([], f)

        # ---------------- Phase 1 ----------------
        print(f"\n[Phase 1 - {task}] Fine-tune full model with PGD")
        for param in model.clip_model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6, weight_decay=1e-3)

        train_loader = make_loader_from_pth(
            f"xxx/{task}_train",
            batch_size=args.batch_size,
            augment=True,
            task_name=task
        )

        for epoch in range(args.pretrain_epochs):
            model.train()
            model.head_dict[task].current_epoch = epoch
            for batch in train_loader:
                optimizer.zero_grad()
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                adv_images = pgd_attack(model, pixel_values, labels, task, alpha=0.53/255, iters=15)

                bsz = pixel_values.size(0)
                adv_count = int(round(args.adv_ratio * bsz))
                adv_count = max(1, min(adv_count, bsz))

                if adv_count >= bsz:
                    mixed_images = adv_images
                    mixed_labels = labels
                else:
                    idxs = list(range(bsz))
                    random.shuffle(idxs)
                    adv_idxs = idxs[:adv_count]
                    orig_idxs = idxs[adv_count:]

                    adv_part = adv_images[adv_idxs]
                    adv_labels_part = labels[adv_idxs]
                    orig_part = pixel_values[orig_idxs]
                    orig_labels_part = labels[orig_idxs]

                    mixed_images = torch.cat([adv_part, orig_part], dim=0)
                    mixed_labels = torch.cat([adv_labels_part, orig_labels_part], dim=0)

                    perm = torch.randperm(mixed_images.size(0))
                    mixed_images = mixed_images[perm]
                    mixed_labels = mixed_labels[perm]

                outputs = model(mixed_images, task_name=task, labels=mixed_labels)
                outputs["loss"].backward()
                optimizer.step()

        # ---------------- Phase 2 ----------------
        print(f"\n[Phase 2 - {task}] Freeze CLIP, train head only")
        for param in model.clip_model.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW(model.head_dict[task].parameters(), lr=1e-4, weight_decay=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        metrics = {
            "train": {"loss": [], "acc": [], "attack_success": []},
            "val_real": {"loss": [], "acc": [], "adv_loss": [], "adv_acc": [], "attack_success": []},
            "val_sdgen": {"loss": [], "acc": [], "adv_loss": [], "adv_acc": [], "attack_success": []},
        }

        for epoch in range(args.epochs):
            model.train()
            model.head_dict[task].current_epoch = epoch
            total_loss, total_correct, total_samples = 0, 0, 0
            total_success, total_batches = 0, 0

            for batch in train_loader:
                optimizer.zero_grad()
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                adv_images = pgd_attack(model, pixel_values, labels, task, alpha=0.53/255, iters=15)
                success = compute_attack_success_rate(model, adv_images, labels, task)
                total_success += success
                total_batches += 1

                # Mix adv and clean samples
                bsz = pixel_values.size(0)
                adv_count = int(round(args.adv_ratio * bsz))
                adv_count = max(1, min(adv_count, bsz))

                if adv_count >= bsz:
                    mixed_images = adv_images
                    mixed_labels = labels
                else:
                    idxs = list(range(bsz))
                    random.shuffle(idxs)
                    adv_idxs = idxs[:adv_count]
                    orig_idxs = idxs[adv_count:]

                    adv_part = adv_images[adv_idxs]
                    adv_labels_part = labels[adv_idxs]
                    orig_part = pixel_values[orig_idxs]
                    orig_labels_part = labels[orig_idxs]

                    mixed_images = torch.cat([adv_part, orig_part], dim=0)
                    mixed_labels = torch.cat([adv_labels_part, orig_labels_part], dim=0)

                    perm = torch.randperm(mixed_images.size(0))
                    mixed_images = mixed_images[perm]
                    mixed_labels = mixed_labels[perm]

                outputs = model(mixed_images, task_name=task, labels=mixed_labels)
                logits = outputs["logits"]
                if task_type_dict[task] == "multiclass":
                    preds = torch.argmax(logits, dim=1)
                else:
                    preds = (torch.sigmoid(logits.squeeze()) > 0.5).long()

                loss = outputs["loss"]
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * mixed_labels.size(0)
                total_correct += (preds == mixed_labels).sum().item()
                total_samples += mixed_labels.size(0)

            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            avg_success = total_success / max(1, total_batches)
            metrics["train"]["loss"].append(avg_loss)
            metrics["train"]["acc"].append(avg_acc)
            metrics["train"]["attack_success"].append(avg_success)

            for domain in ["Real", "SD-Gen"]:
                tag = f"val_{domain.lower().replace('-', '')}"
                val_loader = make_loader_from_pth(
                    f"xxx/{task}_val_{domain.lower().replace('-', '')}",
                    batch_size=args.batch_size,
                    augment=False,
                    task_name=task
                )
                # Clean validation
                loss, acc, _ = evaluate(model, val_loader, task, adv=False)
                metrics[tag]["loss"].append(loss)
                metrics[tag]["acc"].append(acc)
                # Adv validation
                adv_loss, adv_acc, adv_success = evaluate(model, val_loader, task, adv=True, alpha=0.53/255, iters=15)
                metrics[tag]["adv_loss"].append(adv_loss)
                metrics[tag]["adv_acc"].append(adv_acc)
                metrics[tag]["attack_success"].append(adv_success)

            scheduler.step(metrics['val_real']['loss'][-1])

            # Log epoch
            epoch_record = {
                "task": task,
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_acc": avg_acc,
                "train_attack_success": avg_success,
                "val_real_loss": metrics['val_real']['loss'][-1],
                "val_real_acc": metrics['val_real']['acc'][-1],
                "val_real_adv_loss": metrics['val_real']['adv_loss'][-1],
                "val_real_adv_acc": metrics['val_real']['adv_acc'][-1],
                "val_real_attack_success": metrics['val_real']['attack_success'][-1],
                "val_sdgen_loss": metrics['val_sdgen']['loss'][-1],
                "val_sdgen_acc": metrics['val_sdgen']['acc'][-1],
                "val_sdgen_adv_loss": metrics['val_sdgen']['adv_loss'][-1],
                "val_sdgen_adv_acc": metrics['val_sdgen']['adv_acc'][-1],
                "val_sdgen_attack_success": metrics['val_sdgen']['attack_success'][-1],
            }
            with open(log_file_path, "r+") as f:
                logs = json.load(f)
                logs.append(epoch_record)
                f.seek(0)
                json.dump(logs, f, indent=2)
                f.truncate()

        # Plot & save head
        plot_metrics(metrics, args.results_dir, task)
        head_path = os.path.join(args.results_dir, f"{task}_head.pth")
        torch.save(model.head_dict[task].state_dict(), head_path)
        print(f"[✓] Saved classifier head for task '{task}' to {head_path}")

        with open(os.path.join(args.results_dir, f"{task}_logs.json"), "w") as f:
            json.dump({task: metrics}, f, indent=2)

        print(f"\n[✓] {task.capitalize()} task complete. Results saved.")


if __name__ == "__main__":
    main()
