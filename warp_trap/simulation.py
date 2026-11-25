"""Batched, independent 3D particles in isotropic harmonic traps."""

import numpy as np
import warp as wp

METHODS = ("explicit", "semi-implicit")
POSITION = np.array([1.0, -0.5, 0.25], dtype=np.float32)
VELOCITY = np.array([0.2, 0.4, -0.1], dtype=np.float32)


@wp.kernel
def integrate(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    omega: wp.array(dtype=wp.float32),
    dt: float,
    velocity_first: int,
):
    i = wp.tid()
    x = positions[i]
    v = velocities[i]
    w = omega[i]
    next_v = v - dt * w * w * x
    if velocity_first != 0:
        v = next_v
    positions[i] = x + dt * v
    velocities[i] = next_v


def initial_state(count, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(count, 3)).astype(np.float32)
    v = rng.uniform(-1, 1, size=(count, 3)).astype(np.float32)
    omega = rng.uniform(0.5, 2, size=count).astype(np.float32)
    return x, v, omega


class Particles:
    def __init__(self, x, v, omega, device):
        self.device = wp.get_device(device)
        self.count = len(omega)
        self.x = wp.array(x, dtype=wp.vec3, device=self.device)
        self.v = wp.array(v, dtype=wp.vec3, device=self.device)
        self.omega = wp.array(omega, dtype=wp.float32, device=self.device)

    def step(self, method, dt):
        if method not in METHODS:
            raise ValueError(f"Unknown integrator {method!r}")
        wp.launch(
            integrate,
            dim=self.count,
            inputs=[self.x, self.v, self.omega, dt, int(method == "semi-implicit")],
            device=self.device,
        )

    def snapshot(self):
        wp.synchronize_device(self.device)
        return self.x.numpy().astype(np.float64), self.v.numpy().astype(np.float64)


def analytical_state(x0, v0, omega, t):
    """Exact solution for unit masses and positive angular frequencies."""
    w = np.asarray(omega, dtype=np.float64)[..., None]
    x0 = np.asarray(x0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    phase = w * t
    return (
        x0 * np.cos(phase) + v0 / w * np.sin(phase),
        v0 * np.cos(phase) - w * x0 * np.sin(phase),
    )


def energy(x, v, omega):
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(omega, dtype=np.float64)
    return 0.5 * (np.sum(v * v, axis=-1) + w**2 * np.sum(x * x, axis=-1))


def modified_energy(x, v, omega, dt):
    """Quadratic invariant of the velocity-first semi-implicit map."""
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(omega, dtype=np.float64)
    return energy(x, v, w) - 0.5 * dt * w**2 * np.sum(x * v, axis=-1)
