from __future__ import annotations

import torch
from torch import nn


class DenseBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.skip = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values) + self.skip(values)


class SpatialEncoder(nn.Module):
    """Dense XPU-compatible neighbor MLP with masked mean/max pooling."""

    def __init__(
        self,
        focal_dim: int,
        neighbor_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.neighbor_encoder = DenseBlock(neighbor_dim, hidden_dim, dropout)
        self.focal_encoder = DenseBlock(focal_dim, hidden_dim, dropout)
        self.fusion = nn.Sequential(
            DenseBlock(hidden_dim * 3, hidden_dim, dropout),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        focal: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.neighbor_encoder(neighbors)
        mask = neighbor_mask.unsqueeze(-1)
        masked = encoded * mask
        counts = mask.sum(dim=1).clamp_min(1)
        mean_pool = masked.sum(dim=1) / counts
        minimum = torch.finfo(encoded.dtype).min
        max_pool = encoded.masked_fill(~mask, minimum).max(dim=1).values
        has_neighbor = neighbor_mask.any(dim=1, keepdim=True)
        max_pool = torch.where(has_neighbor, max_pool, torch.zeros_like(max_pool))
        focal_encoded = self.focal_encoder(focal)
        return self.fusion(torch.cat([focal_encoded, mean_pool, max_pool], dim=-1))


def masked_huber_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = nn.functional.huber_loss(prediction, target, reduction="none")
    selected = losses[mask]
    if selected.numel() == 0:
        return prediction.sum() * 0
    return selected.mean()


class SpatialMaskedAutoencoder(nn.Module):
    """Spatial MAE that reconstructs a focal sector through learned indicator queries."""

    def __init__(
        self,
        focal_dim: int,
        output_dim: int,
        *,
        width: int = 128,
        heads: int = 8,
        encoder_layers: int = 6,
        decoder_layers: int = 4,
        ff_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if width % heads:
            raise ValueError("Transformer width must be divisible by its head count")
        self.focal_dim = focal_dim
        self.output_dim = output_dim
        self.focal_projection = nn.Linear(focal_dim, width)
        self.neighbor_2010_projection = nn.Linear(focal_dim, width)
        self.neighbor_2022_projection = nn.Linear(output_dim * 2, width)
        self.role_embedding = nn.Embedding(3, width)
        encoder_layer = nn.TransformerEncoderLayer(
            width,
            heads,
            dim_feedforward=width * ff_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, encoder_layers, norm=nn.LayerNorm(width))
        decoder_layer = nn.TransformerDecoderLayer(
            width,
            heads,
            dim_feedforward=width * ff_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, decoder_layers, norm=nn.LayerNorm(width))
        self.indicator_queries = nn.Parameter(torch.randn(1, output_dim, width) * 0.02)
        self.output = nn.Linear(width, 1)

    def forward(
        self,
        focal: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, count, _ = neighbors.shape
        neighbor_2010 = neighbors[..., : self.focal_dim]
        neighbor_values = neighbors[..., self.focal_dim : self.focal_dim + self.output_dim]
        visibility = neighbors[
            ..., self.focal_dim + self.output_dim : self.focal_dim + self.output_dim * 2
        ] > 0.5
        neighbor_values = torch.where(visibility, neighbor_values, torch.zeros_like(neighbor_values))
        visible_sector = visibility.any(dim=-1) & neighbor_mask

        focal_token = self.focal_projection(focal).unsqueeze(1) + self.role_embedding.weight[0].view(1, 1, -1)
        context_2010 = self.neighbor_2010_projection(neighbor_2010) + self.role_embedding.weight[1].view(1, 1, -1)
        value_features = torch.cat([neighbor_values, visibility.to(neighbor_values.dtype)], dim=-1)
        context_2022 = self.neighbor_2022_projection(value_features) + self.role_embedding.weight[2].view(1, 1, -1)
        memory_input = torch.cat([focal_token, context_2010, context_2022], dim=1)
        padding = torch.cat(
            [
                torch.zeros((batch, 1), dtype=torch.bool, device=focal.device),
                ~neighbor_mask,
                ~visible_sector,
            ],
            dim=1,
        )
        memory = self.encoder(memory_input, src_key_padding_mask=padding)
        queries = self.indicator_queries.expand(batch, -1, -1)
        decoded = self.decoder(queries, memory, memory_key_padding_mask=padding)
        return self.output(decoded).squeeze(-1)


class GATReconstructor(nn.Module):
    """Learned 2010-to-2022 change model with optional visible-neighbor correction."""

    def __init__(
        self,
        focal_dim: int,
        neighbor_dim: int,
        output_dim: int,
        *,
        width: int = 128,
        heads: int = 4,
        dropout: float = 0.05,
        context_dropout: float = 0.25,
        indicator_dropout: float = 0.10,
    ):
        super().__init__()
        if width % heads:
            raise ValueError("GAT width must be divisible by its head count")
        context_dim = output_dim * 2 + 3
        neighbor_focal_dim = neighbor_dim - context_dim
        if neighbor_focal_dim != focal_dim:
            raise ValueError(
                f"GAT expects neighbor_dim=focal_dim+2*output_dim+3; received {neighbor_dim}"
            )
        self.focal_dim = focal_dim
        self.output_dim = output_dim
        self.context_dropout = context_dropout
        self.indicator_dropout = indicator_dropout
        self.focal_encoder = DenseBlock(focal_dim, width, dropout)
        self.neighbor_2010_encoder = DenseBlock(focal_dim + 3, width, dropout)
        self.neighbor_change_encoder = DenseBlock(focal_dim + output_dim * 2 + 3, width, dropout)
        self.focal_query = nn.Linear(width, width)
        self.all_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.visible_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.base_fusion = DenseBlock(width * 2, width, dropout)
        self.correction_fusion = DenseBlock(width * 3, width, dropout)
        self.base_output = nn.Linear(width, output_dim)
        self.correction_output = nn.Linear(width, output_dim)
        self.gate = nn.Sequential(nn.Linear(width + 2, width), nn.GELU(), nn.Linear(width, output_dim))
        nn.init.zeros_(self.base_output.weight)
        nn.init.zeros_(self.base_output.bias)
        nn.init.zeros_(self.correction_output.weight)
        nn.init.zeros_(self.correction_output.bias)

    def _augment_visibility(
        self, changes: torch.Tensor, visibility: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return changes, visibility
        batch = changes.shape[0]
        rates = changes.new_tensor([0.0, 0.10, 0.20, 0.30])
        selected_rates = rates[torch.randint(0, len(rates), (batch,), device=changes.device)].view(batch, 1, 1)
        keep = torch.rand_like(changes) >= selected_rates
        if self.indicator_dropout:
            keep &= torch.rand_like(changes) >= self.indicator_dropout
        if self.context_dropout:
            keep_context = torch.rand((batch, 1, 1), device=changes.device) >= self.context_dropout
            keep &= keep_context
        visibility = visibility & keep
        return torch.where(visibility, changes, torch.zeros_like(changes)), visibility

    @staticmethod
    def _attend(
        attention: nn.MultiheadAttention,
        query: torch.Tensor,
        tokens: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        has_valid = valid.any(dim=1, keepdim=True)
        safe_valid = valid.clone()
        if safe_valid.shape[1]:
            safe_valid[:, 0] |= ~has_valid.squeeze(1)
        attended = attention(query, tokens, tokens, key_padding_mask=~safe_valid, need_weights=False)[0].squeeze(1)
        return torch.where(has_valid, attended, torch.zeros_like(attended))

    def forward(
        self,
        focal: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        neighbor_focal = neighbors[..., : self.focal_dim]
        start = self.focal_dim
        changes = neighbors[..., start : start + self.output_dim]
        visibility = neighbors[..., start + self.output_dim : start + self.output_dim * 2] > 0.5
        relative = neighbors[..., -3:]
        changes, visibility = self._augment_visibility(changes, visibility)
        visible_sector = visibility.any(dim=-1) & neighbor_mask

        focal_encoded = self.focal_encoder(focal)
        query = self.focal_query(focal_encoded).unsqueeze(1)
        all_tokens = self.neighbor_2010_encoder(torch.cat([neighbor_focal, relative], dim=-1))
        visible_tokens = self.neighbor_change_encoder(
            torch.cat([neighbor_focal, changes, visibility.to(changes.dtype), relative], dim=-1)
        )
        all_context = self._attend(self.all_attention, query, all_tokens, neighbor_mask)
        visible_context = self._attend(self.visible_attention, query, visible_tokens, visible_sector)
        base_features = self.base_fusion(torch.cat([focal_encoded, all_context], dim=-1))
        correction_features = self.correction_fusion(
            torch.cat([focal_encoded, all_context, visible_context], dim=-1)
        )
        neighbor_count = neighbor_mask.sum(dim=1, keepdim=True).clamp_min(1)
        visible_fraction = visible_sector.sum(dim=1, keepdim=True) / neighbor_count
        has_visible = visible_sector.any(dim=1, keepdim=True)
        gate_features = torch.cat(
            [base_features, visible_fraction.to(focal.dtype), has_visible.to(focal.dtype)], dim=-1
        )
        gate = torch.sigmoid(self.gate(gate_features)) * has_visible.to(focal.dtype)
        return self.base_output(base_features) + gate * self.correction_output(correction_features)


def build_reconstruction_model(
    architecture: str,
    focal_dim: int,
    neighbor_dim: int,
    output_dim: int,
    params: dict[str, float | int],
    architecture_config: dict[str, float | int],
) -> nn.Module:
    if architecture == "spatial":
        return SpatialEncoder(
            focal_dim,
            neighbor_dim,
            output_dim,
            hidden_dim=int(params["hidden_width"]),
            dropout=float(params["dropout"]),
        )
    if architecture == "mae":
        expected = focal_dim + output_dim * 2
        if neighbor_dim != expected:
            raise ValueError(f"MAE expects {expected} neighbor features, received {neighbor_dim}")
        return SpatialMaskedAutoencoder(
            focal_dim,
            output_dim,
            width=int(params["width"]),
            heads=int(architecture_config.get("heads", 8)),
            encoder_layers=int(architecture_config.get("encoder_layers", 6)),
            decoder_layers=int(architecture_config.get("decoder_layers", 4)),
            ff_ratio=int(architecture_config.get("ff_ratio", 4)),
            dropout=float(params["dropout"]),
        )
    if architecture == "gat":
        return GATReconstructor(
            focal_dim,
            neighbor_dim,
            output_dim,
            width=int(params["width"]),
            heads=int(architecture_config.get("heads", 4)),
            dropout=float(params["dropout"]),
            context_dropout=float(architecture_config.get("context_dropout", 0.25)),
            indicator_dropout=float(architecture_config.get("indicator_dropout", 0.10)),
        )
    raise ValueError(f"Unknown reconstruction architecture: {architecture}")
