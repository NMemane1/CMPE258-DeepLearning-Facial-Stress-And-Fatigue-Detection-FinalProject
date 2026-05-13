"""Tests for model architecture. These run on CPU."""
import pytest
import torch
import torch.nn as nn

# Skip the whole module if heavy deps aren't installed
transformers = pytest.importorskip("transformers")


class TestHeads:
    def test_shared_mlp_forward(self):
        from src.models.heads import SharedMLP
        mlp = SharedMLP(in_dim=384, hidden=256, dropout=0.3)
        x = torch.randn(4, 384)
        y = mlp(x)
        assert y.shape == (4, 256)

    def test_classification_head_shape(self):
        from src.models.heads import ClassificationHead
        head = ClassificationHead(in_dim=256, num_classes=3, dropout=0.3)
        head.train()  # BN needs train mode for batch > 1
        x = torch.randn(8, 256)
        y = head(x)
        assert y.shape == (8, 3)

    def test_multitask_heads(self):
        from src.models.heads import MultiTaskHeads
        m = MultiTaskHeads(384, 256, 3, 2, dropout=0.2)
        m.train()
        x = torch.randn(4, 384)
        out = m(x)
        assert out.stress_logits.shape == (4, 3)
        assert out.fatigue_logits.shape == (4, 2)
        assert out.shared_features.shape == (4, 256)


class TestLosses:
    def test_focal_loss_ignores_minus_one(self):
        from src.training.losses import FocalLoss
        fl = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 3)
        # all -1 labels → zero loss
        tgt = torch.tensor([-1, -1, -1, -1], dtype=torch.long)
        loss = fl(logits, tgt)
        assert loss.item() == 0.0

    def test_focal_loss_nonzero_with_valid_labels(self):
        from src.training.losses import FocalLoss
        fl = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 3)
        tgt = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        loss = fl(logits, tgt)
        assert loss.item() > 0

    def test_masked_ce_ignores_minus_one(self):
        from src.training.losses import MaskedCrossEntropy
        ce = MaskedCrossEntropy()
        logits = torch.randn(4, 3)
        tgt = torch.tensor([0, -1, 1, -1], dtype=torch.long)
        loss = ce(logits, tgt)
        assert loss.item() > 0

    def test_multitask_loss_combines(self):
        from src.training.losses import MultiTaskLoss
        mt = MultiTaskLoss(weight_focal=0.5)
        s_logits = torch.randn(4, 3)
        f_logits = torch.randn(4, 2)
        s_tgt = torch.tensor([0, 1, 2, -1], dtype=torch.long)
        f_tgt = torch.tensor([-1, -1, 0, 1], dtype=torch.long)
        out = mt(s_logits, f_logits, s_tgt, f_tgt)
        assert out.total.item() > 0
        assert out.stress_ce.item() >= 0
        assert out.fatigue_ce.item() >= 0
