# AuxClassifier/sequence_classifier.py

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    标准正弦位置编码。
    输入/输出形状: [B, T, C]
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, T, C]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class MaskedAttentionPooling(nn.Module):
    """
    带 mask 的注意力池化。
    输入:
        x:    [B, T, C]
        mask: [B, T]，1 表示有效帧，0 表示 padding
    输出:
        pooled: [B, C]
        attn:   [B, T]
    """

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, C]
        logits = self.score(x).squeeze(-1)  # [B, T]

        if mask is not None:
            mask = SequenceClassifier.normalize_mask(mask, x.size(0), x.size(1), x.device)
            logits = logits.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(logits, dim=1)  # [B, T]

        # 避免全无效帧时出现 nan
        if mask is not None:
            attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)

        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)  # [B, C]
        return pooled, attn


class SequenceClassifier(nn.Module):
    """
    通用时序分类器，供情感分类 / 身份风格分类共用。

    设计目标：
    1. 直接输入 exp+jaw 的序列特征；
    2. 支持 mask；
    3. 支持返回中间特征，便于后续交换实验做距离类指标；
    4. 独立于原主模型，不影响现有训练和采样代码。

    推荐输入：
        x = torch.cat([exp, jaw], dim=-1)  # [B, T, 103]

    也支持：
        logits = model.forward_from_exp_jaw(exp, jaw, mask)
    """

    def __init__(
        self,
        input_dim: int = 103,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 512,
        num_classes: int = 8,
        dropout: float = 0.1,
        max_len: int = 128,
        use_cls_token: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_cls_token = use_cls_token

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_encoding = PositionalEncoding(
            d_model=hidden_dim,
            max_len=max_len + 1 if use_cls_token else max_len,
            dropout=dropout,
        )

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        else:
            self.cls_token = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        self.attn_pool = MaskedAttentionPooling(hidden_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.reset_parameters()

    def reset_parameters(self):
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def normalize_mask(
        mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        将 mask 统一成 bool 类型的 [B, T]。
        支持输入:
            [B, T]
            [B, 1, T]
            [T]
        """
        if mask is None:
            return None

        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(batch_size, -1)
        elif mask.dim() == 3 and mask.size(1) == 1:
            mask = mask.squeeze(1)

        if mask.dim() != 2:
            raise ValueError(f"mask shape should be [B,T] / [B,1,T] / [T], but got {mask.shape}")

        mask = mask.to(device)
        if mask.dtype != torch.bool:
            mask = mask > 0

        if mask.size(0) != batch_size or mask.size(1) != seq_len:
            raise ValueError(
                f"mask shape mismatch, expect [{batch_size}, {seq_len}], got {list(mask.shape)}"
            )

        return mask

    def extract_features(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        """
        提取时序特征。
        输入:
            x: [B, T, F]
            mask: [B, T]，1 表示有效帧
        输出:
            pooled: [B, C]
            （可选）attn: [B, T] 或 cls_token attention 的替代结果
        """
        if x.dim() != 3:
            raise ValueError(f"x must be [B, T, F], but got shape {x.shape}")

        b, t, _ = x.shape
        device = x.device
        mask = self.normalize_mask(mask, b, t, device)

        x = self.input_proj(x)  # [B, T, C]

        if self.use_cls_token:
            cls_token = self.cls_token.expand(b, -1, -1)  # [B, 1, C]
            x = torch.cat([cls_token, x], dim=1)          # [B, T+1, C]

            if mask is not None:
                cls_mask = torch.ones((b, 1), dtype=torch.bool, device=device)
                mask = torch.cat([cls_mask, mask], dim=1)  # [B, T+1]

        x = self.pos_encoding(x)

        # Transformer 的 src_key_padding_mask: True 表示 padding，需要被忽略
        key_padding_mask = None if mask is None else ~mask
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        if self.use_cls_token:
            pooled = x[:, 0]  # [B, C]
            if return_attn:
                # cls token 模式下不额外输出 attention pool 权重
                return pooled, None
            return pooled

        pooled, attn = self.attn_pool(x, mask=mask)
        if return_attn:
            return pooled, attn
        return pooled

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
        return_attn: bool = False,
    ):
        """
        Args:
            x: [B, T, F]
            mask: [B, T]
            return_features: 是否返回 pooled feature
            return_attn: 是否返回池化权重（仅在 use_cls_token=False 时有效）

        Returns:
            logits
            或 logits, features
            或 logits, features, attn
        """
        if return_attn:
            features, attn = self.extract_features(x, mask=mask, return_attn=True)
        else:
            features = self.extract_features(x, mask=mask, return_attn=False)
            attn = None

        logits = self.head(features)

        if return_features and return_attn:
            return logits, features, attn
        elif return_features:
            return logits, features
        elif return_attn:
            return logits, attn
        return logits

    def forward_from_exp_jaw(
        self,
        exp: torch.Tensor,
        jaw: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
        return_attn: bool = False,
    ):
        """
        直接输入 exp 和 jaw。
        exp: [B, T, D_exp]
        jaw: [B, T, D_jaw]
        """
        if exp.dim() != 3 or jaw.dim() != 3:
            raise ValueError(
                f"exp and jaw must both be [B, T, D], but got exp={exp.shape}, jaw={jaw.shape}"
            )
        if exp.size(0) != jaw.size(0) or exp.size(1) != jaw.size(1):
            raise ValueError(
                f"exp and jaw batch/time mismatch, got exp={exp.shape}, jaw={jaw.shape}"
            )

        x = torch.cat([exp, jaw], dim=-1)
        return self.forward(
            x,
            mask=mask,
            return_features=return_features,
            return_attn=return_attn,
        )

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = self.forward(x, mask=mask)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        probs = self.predict_proba(x, mask=mask)
        return torch.argmax(probs, dim=-1)