import torch

from nbody_ctm.models import NBodyCTMClassifier
from nbody_ctm.losses import ctm_classification_loss


def test_ctm_shapes_and_backward():
    enc = dict(type="graph_temporal", d_hidden=24, d_out=32, temporal_blocks=1, graph_blocks=1, temporal_kernel=3, temporal_stride=2, dropout=0.0)
    ctm = dict(iterations=4, d_model=40, d_input=32, heads=4, n_synch_out=32, n_synch_action=32, synapse_depth=2, memory_length=4, memory_hidden_dims=10, dropout=0.0, dropout_nlm=0.0, deep_nlms=True, n_random_pairing_self=4, pairing_seed=1)
    model = NBodyCTMClassifier(enc, ctm, n_classes=3)
    state = torch.randn(3, 12, 5, 4)
    mass = torch.randn(3, 5)
    pred, cert, tracking = model(state, mass, track=True)
    assert pred.shape == (3, 3, 4)
    assert cert.shape == (3, 4)
    assert tracking["post_activations"].shape == (3, 4, 40)
    assert tracking["attention"].shape[1] == 4
    y = torch.tensor([0, 1, 2])
    loss, pieces = ctm_classification_loss(pred, cert, y, final_ce_weight=0.5)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
