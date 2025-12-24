import os
import warnings

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from transformers import Trainer

import torch.nn.functional as F
from torch.nn import MultiheadAttention
import math

# Suppress all warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(data_dir, processor, batch_size=32):
    """
    Loads image data using torchvision's ImageFolder and applies the processor.

    Args:
        data_dir (str): Directory containing image data.
        processor (callable): Function or processor for image preprocessing.
        batch_size (int): Batch size for DataLoader.

    Returns:
        DataLoader: DataLoader for processed images.
    """
    dataset = datasets.ImageFolder(root=data_dir)
    processed_dataset = CustomImageFolderDataset(dataset, processor)
    dataloader = DataLoader(processed_dataset, batch_size=batch_size, num_workers=32, shuffle=False)
    return dataloader


class CustomTrainer(Trainer):
    """
    Custom Trainer that saves only the classifier head parameters.
    """

    def save_model(self, output_dir=None, **kwargs):
        """
        Overrides the save_model method to save only the classifier head.
        """
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        classifier_head_path = os.path.join(output_dir, "classifier_head.pth")
        torch.save(self.model.classifier_head.state_dict(), classifier_head_path)

        # Optionally save processor if available
        if hasattr(self, "processor"):
            self.processor.save_pretrained(output_dir)
        # Do not call parent's save_model (prevents saving the full model)


class CustomImageFolderDataset(Dataset):
    """
    Custom dataset that wraps torchvision's ImageFolder and applies a processor.
    """

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            image, label = self.dataset[idx]
        except (OSError, IOError) as e:
            print(f"Error loading image: {self.dataset.imgs[idx][0]} - {e}")
            # Skip problematic image and move to next
            return self.__getitem__((idx + 1) % len(self.dataset))
        processed_image = self.processor(images=image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": processed_image.squeeze(), "labels": label}

class AdvancedClassifierHead_CLIP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, dropout_rate=0.3, num_classes=10):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        attention_weights = self.attention(x)
        x = x * attention_weights
        logits = self.classifier(x)  # shape: [batch, num_classes]
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return {"logits": logits, "loss": loss}
        else:
            return {"logits": logits}


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=7, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)  # 多分类输出
        )

    def forward(self, x):
        return self.net(x)

def gram_matrix(x):
    """
    输入: x (B, D)
    输出: Gram 矩阵 (B, D, D)
    """
    B, D = x.size()
    x = x.unsqueeze(2)              # (B, D, 1)
    gram = torch.bmm(x, x.transpose(1, 2))  # (B, D, D)
    return gram / D                 # 归一化
  
class Multi_MultiC_GramCluster_v2(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, num_classes=10, dropout_rate=0.3, gram_reduce_dim=512, cluster_factor=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.gram_reduce_dim = gram_reduce_dim
        self.cluster_factor = cluster_factor

        # 主干,通道注意力机制: 学习输入特征的权重分布，强调重要特征
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()        #生成0-1的注意力权重

        )
        #前馈网络:提供非线性变换能力
        self.feedforward = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim * 4, input_dim)
        )
        #进一步的特征处理层
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        #更深层的特征提取网络
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.output = nn.Linear(hidden_dim // 2, num_classes)

        # Gram 降维
        self.gram_reduce = nn.Linear(input_dim * input_dim, gram_reduce_dim)
        #将主干特征和Gram聚类特征融合
        self.gram_mlp = nn.Sequential(
            nn.Linear(input_dim + gram_reduce_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # 聚类中心
        self.register_buffer("cluster_centers", torch.zeros(num_classes * cluster_factor, gram_reduce_dim))
        self.cluster_initialized = False

    def compute_gram(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W)
        G = torch.bmm(x_flat, x_flat.transpose(1,2)) / (H*W)
        G_flat = G.view(B, C*C)
        G_red = self.gram_reduce(G_flat)  # 降维
        return G_red  # [B, gram_reduce_dim]

    def initialize_clusters(self, gram_features):
        # gram_features: [N, gram_reduce_dim]
        k = self.num_classes * self.cluster_factor
        kmeans = KMeans(n_clusters=k, random_state=42).fit(gram_features.cpu().numpy())
        centers = torch.tensor(kmeans.cluster_centers_, device=gram_features.device, dtype=gram_features.dtype)
        self.cluster_centers.copy_(centers)
        self.cluster_initialized = True

    def forward(self, x_embed, x_image=None, use_gram=True):
        """
        x_embed: [B, input_dim] 主干 embedding
        x_image: [B, C, H, W] 原始图像，仅用于 Gram 提取
        """
        x_attn = x_embed * self.attn(x_embed)  # 通道注意力, 原始特征 × 注意力权重
        x_ffn = self.feedforward(x_attn)
        x_main = x_attn + x_ffn  # 残差

        if use_gram and x_image is not None:
            gram_feat = self.compute_gram(x_image)  # [B, gram_reduce_dim]
            if not self.cluster_initialized:
                # 初始化聚类中心
                self.initialize_clusters(gram_feat)
            # 找最近中心
            dist = torch.cdist(gram_feat, self.cluster_centers)  # [B, K]计算欧氏距离
            nearest_idx = dist.argmin(dim=1)
            cluster_feat = self.cluster_centers[nearest_idx]  # [B, gram_reduce_dim]
            # 融合
            x_fused = torch.cat([x_main, cluster_feat], dim=1)
            x_h = self.gram_mlp(x_fused)
        else:
            x_h = x_main

        h = self.mlp1(x_h)
        h = h + x_h
        h = self.mlp2(h)
        logits = self.output(h)
        return logits

class CombinedModel(nn.Module):
    """
    End-to-end model combining a ViT encoder and an advanced classifier head.
    Used for binary classification tasks.
    """

    def __init__(self, vit_model, classifier_head):
        super().__init__()
        self.vit = vit_model.eval()
        self.classifier_head = classifier_head
        self.loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss

    def forward(self, pixel_values, labels=None):
        # Extract [CLS] token output from ViT
        vit_outputs = self.vit(pixel_values, output_hidden_states=True)
        cls_token_output = vit_outputs.hidden_states[-1][:, 0, :]  # [CLS] token
        logits = self.classifier_head(cls_token_output)
        loss = None
        if labels is not None:
            labels = labels.float()  # Ensure labels are float for BCE loss
            loss = self.loss_fn(logits.squeeze(), labels)
        return {"logits": logits, "loss": loss}

    def predict(self, pixel_values):
        vit_outputs = self.vit(pixel_values, output_hidden_states=True)
        cls_token_output = vit_outputs.hidden_states[-1][:, 0, :]
        logits = self.classifier_head(cls_token_output)
        predictions = torch.sigmoid(logits).squeeze().round().cpu().numpy()
        return predictions


class CLIPBinaryClassifier(nn.Module):
    """
    Binary classifier using frozen CLIP image features and an advanced classifier head.
    """

    def __init__(self, clip_model, classifier_head):
        super(CLIPBinaryClassifier, self).__init__()
        self.clip_model = clip_model
        self.classifier_head = classifier_head
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pixel_values, labels=None):
        with torch.no_grad():
            # Freeze the CLIP image encoder during training
            clip_outputs = self.clip_model.get_image_features(pixel_values)
        logits = self.classifier_head(clip_outputs)
        loss = None
        if labels is not None:
            labels = labels.float()
            loss = self.loss_fn(logits.squeeze(), labels)
        return {"logits": logits, "loss": loss}

    def predict(self, pixel_values):
        with torch.no_grad():
            clip_outputs = self.clip_model.get_image_features(pixel_values)
        logits = self.classifier_head(clip_outputs)
        preds = torch.sigmoid(torch.tensor(logits)).squeeze().round().cpu().numpy()
        return preds
class MultiHeadCLIPClassifier(nn.Module):
    def __init__(self, clip_model, head_dict, task_type_dict):
        super().__init__()
        self.clip_model = clip_model
        self.head_dict = nn.ModuleDict(head_dict)
        self.task_type_dict = task_type_dict

    def forward(self, pixel_values, task_name, labels=None, detach_clip=True):
        # detach_clip=True 用于 Phase 2 冻结 CLIP
        if detach_clip:
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(pixel_values)
        else:
            image_features = self.clip_model.get_image_features(pixel_values)

        # 归一化
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # 分类头
        head_output = self.head_dict[task_name](image_features)
        logits = head_output["logits"] if isinstance(head_output, dict) else head_output

        task_type = self.task_type_dict[task_name]
        if task_type == "multiclass":
            if logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            loss = F.cross_entropy(logits, labels) if labels is not None else None
        else:  # binary
            logits = logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels.float()) if labels is not None else None

        return {"logits": logits, "loss": loss}
