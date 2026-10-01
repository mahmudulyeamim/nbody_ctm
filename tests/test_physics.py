import torch

from nbody_ctm.physics import accelerations, total_energy, velocity_verlet_step


def test_two_body_acceleration_is_opposite_weighted_by_mass():
    pos = torch.tensor([[[-1.0, 0.0], [1.0, 0.0]]])
    vel = torch.zeros_like(pos)
    mass = torch.tensor([[1.0, 2.0]])
    a = accelerations(pos, mass, softening=0.0)
    assert a[0, 0, 0] > 0
    assert a[0, 1, 0] < 0
    # momentum derivative sums to zero
    assert torch.allclose((a * mass[..., None]).sum(1), torch.zeros(1, 2), atol=1e-6)


def test_verlet_short_energy_conservation():
    pos = torch.tensor([[[-1.0, 0.0], [1.0, 0.0]]], dtype=torch.float64)
    mass = torch.ones(1, 2, dtype=torch.float64)
    # equal-mass circular-ish binary in G=1 units
    speed = (1.0 / 4.0) ** 0.5
    vel = torch.tensor([[[0.0, speed], [0.0, -speed]]], dtype=torch.float64)
    e0 = total_energy(pos, vel, mass, softening=1e-4)
    for _ in range(200):
        pos, vel = velocity_verlet_step(pos, vel, mass, dt=0.002, G=1.0, softening=1e-4)
    e1 = total_energy(pos, vel, mass, softening=1e-4)
    rel = ((e1 - e0).abs() / e0.abs()).item()
    assert rel < 2e-3
