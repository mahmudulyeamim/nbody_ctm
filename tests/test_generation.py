import torch

from nbody_ctm.generator import sample_initial_conditions
from nbody_ctm.physics import simulate_and_label


def test_simulation_output_and_no_nan():
    cfg = dict(
        n_bodies=6, G=1.0, softening=0.045, collision_radius=0.035,
        obs_steps=8, hidden_steps=12, dt=0.02, escape_radius=5.5,
        escape_energy_margin=0.03, escape_radial_velocity=0.02,
        escape_persistence=2, guard_steps=2, bound_radius_fraction=0.98,
        bound_collision_clearance=1.1, bound_energy_ceiling=-10.0,
        proposal_probs=[0.7, 0.1, 0.15, 0.05],
    )
    g = torch.Generator().manual_seed(3)
    pos, vel, mass, proposal, q = sample_initial_conditions(8, cfg, torch.device("cpu"), g)
    keys = ["obs_steps", "hidden_steps", "dt", "G", "softening", "collision_radius", "escape_radius", "escape_energy_margin", "escape_radial_velocity", "escape_persistence", "guard_steps", "bound_radius_fraction", "bound_collision_clearance", "bound_energy_ceiling"]
    traj, summary, diag = simulate_and_label(pos, vel, mass, **{k: cfg[k] for k in keys})
    assert traj.shape == (8, 8, 6, 4)
    assert torch.isfinite(traj).all()
    assert summary.label.shape == (8,)
    assert torch.isfinite(diag["initial_total_energy"]).all()
