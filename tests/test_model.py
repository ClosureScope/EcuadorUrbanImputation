from __future__ import annotations

import pytest
import torch

from ecuador_evolution.model import (
    GATReconstructor,
    SpatialEncoder,
    SpatialMaskedAutoencoder,
    masked_huber_loss,
)


def test_spatial_encoder_cpu() -> None:
    model = SpatialEncoder(5, 4, 3, hidden_dim=8, dropout=0)
    focal = torch.randn(6, 5)
    neighbors = torch.randn(6, 12, 4)
    mask = torch.zeros(6, 12, dtype=torch.bool); mask[:, :2] = True
    output = model(focal, neighbors, mask)
    assert output.shape == (6, 3)
    assert torch.isfinite(masked_huber_loss(output, torch.zeros_like(output), torch.ones_like(output, dtype=torch.bool)))


def test_mae_queries_empty_neighbors_and_backward() -> None:
    model = SpatialMaskedAutoencoder(
        5, 3, width=8, heads=2, encoder_layers=1, decoder_layers=1, ff_ratio=2, dropout=0
    )
    focal = torch.randn(4, 5)
    neighbors = torch.randn(4, 2, 11)
    neighbors[..., 8:] = 0
    mask = torch.zeros(4, 2, dtype=torch.bool)
    output = model(focal, neighbors, mask)
    assert output.shape == (4, 3)
    loss = masked_huber_loss(output, torch.randn_like(output), torch.ones_like(output, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_mae_hidden_neighbor_values_are_invariant() -> None:
    model = SpatialMaskedAutoencoder(
        5, 3, width=8, heads=2, encoder_layers=1, decoder_layers=1, ff_ratio=2, dropout=0
    ).eval()
    focal = torch.randn(2, 5)
    neighbors = torch.randn(2, 2, 11)
    neighbors[..., 8:] = 0
    changed = neighbors.clone(); changed[..., 5:8] = torch.tensor([999.0, -999.0, 1234.0])
    mask = torch.ones(2, 2, dtype=torch.bool)
    with torch.no_grad():
        before = model(focal, neighbors, mask)
        after = model(focal, changed, mask)
    assert torch.equal(before, after)


def test_gat_handles_hidden_and_empty_context() -> None:
    model = GATReconstructor(
        focal_dim=10,
        neighbor_dim=10 + 6 + 3,
        output_dim=3,
        width=16,
        heads=4,
        dropout=0,
        context_dropout=0,
        indicator_dropout=0,
    ).eval()
    focal = torch.randn(4, 10)
    neighbors = torch.randn(4, 5, 19)
    neighbors[..., 13:16] = 0
    changed = neighbors.clone(); changed[..., 10:13] = 10_000
    mask = torch.zeros(4, 5, dtype=torch.bool)
    with torch.no_grad():
        before = model(focal, neighbors, mask)
        after = model(focal, changed, mask)
    assert before.shape == (4, 3)
    assert torch.equal(before, after)
    assert torch.isfinite(before).all()


def test_gat_backward_with_visible_context() -> None:
    model = GATReconstructor(10, 19, 3, width=16, heads=4, dropout=0)
    focal = torch.randn(4, 10)
    neighbors = torch.randn(4, 5, 19)
    neighbors[..., 13:16] = 1
    output = model(focal, neighbors, torch.ones(4, 5, dtype=torch.bool))
    loss = output.square().mean(); loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.cuda
def test_spatial_encoder_cuda_smoke() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device("cuda")
    model = SpatialEncoder(5, 4, 3, hidden_dim=8).to(device)
    output = model(torch.randn(2, 5, device=device), torch.randn(2, 12, 4, device=device), torch.ones(2, 12, dtype=torch.bool, device=device))
    assert output.shape == (2, 3)


@pytest.mark.cuda
def test_mae_cuda_forward_backward() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device("cuda")
    model = SpatialMaskedAutoencoder(
        5, 3, width=8, heads=2, encoder_layers=1, decoder_layers=1, ff_ratio=2, dropout=0
    ).to(device)
    neighbors = torch.randn(2, 2, 11, device=device)
    neighbors[..., 8:] = 1
    output = model(
        torch.randn(2, 5, device=device), neighbors, torch.ones(2, 2, dtype=torch.bool, device=device)
    )
    loss = output.square().mean(); loss.backward()
    assert output.shape == (2, 3)
    assert torch.isfinite(loss)
