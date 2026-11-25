"""Differentiable system identification with Warp autodiff.

Each particle is a damped harmonic oscillator with unknown natural frequency
omega and damping ratio zeta:

    dx/dt = v,    dv/dt = -omega^2 x - 2 zeta omega v

We observe noisy positions from a "ground-truth" simulation and recover
(omega, zeta) for every particle at once by differentiating a trajectory-matching
loss through the semi-implicit integrator with ``wp.Tape`` and running Adam.
The gradients are checked against central finite differences.
"""

import numpy as np
import warp as wp


@wp.kernel
def damped_step(
    x_in: wp.array(dtype=wp.vec3),
    v_in: wp.array(dtype=wp.vec3),
    log_omega: wp.array(dtype=wp.float32),
    log_zeta: wp.array(dtype=wp.float32),
    dt: float,
    x_out: wp.array(dtype=wp.vec3),
    v_out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    w = wp.exp(log_omega[i])
    z = wp.exp(log_zeta[i])
    x = x_in[i]
    v = v_in[i]
    v_next = v + dt * (-w * w * x - 2.0 * z * w * v)
    v_out[i] = v_next
    x_out[i] = x + dt * v_next


@wp.kernel
def trajectory_loss(
    x_sim: wp.array(dtype=wp.vec3),
    x_obs: wp.array(dtype=wp.vec3),
    scale: float,
    loss: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    d = x_sim[i] - x_obs[i]
    wp.atomic_add(loss, 0, scale * wp.dot(d, d))


class DampedTrajectory:
    """Unrolled, differentiable rollout with one state buffer per step."""

    def __init__(self, x0, v0, steps, dt, device, requires_grad=True):
        self.device = wp.get_device(device)
        self.count = len(x0)
        self.steps = steps
        self.dt = float(dt)
        kw = dict(dtype=wp.vec3, device=self.device, requires_grad=requires_grad)
        self.x = [wp.array(x0, **kw)] + [wp.zeros(self.count, **kw) for _ in range(steps)]
        self.v = [wp.array(v0, **kw)] + [wp.zeros(self.count, **kw) for _ in range(steps)]

    def rollout(self, log_omega, log_zeta):
        for t in range(self.steps):
            wp.launch(
                damped_step,
                dim=self.count,
                inputs=[self.x[t], self.v[t], log_omega, log_zeta, self.dt],
                outputs=[self.x[t + 1], self.v[t + 1]],
                device=self.device,
            )
        return self.x

    def positions(self):
        return np.stack([x.numpy() for x in self.x[1:]])


def make_observations(count, steps, dt, noise, device, seed=7):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1, 1, size=(count, 3)).astype(np.float32)
    v0 = rng.uniform(-1, 1, size=(count, 3)).astype(np.float32)
    omega = rng.uniform(0.5, 2.0, size=count).astype(np.float32)
    zeta = rng.uniform(0.02, 0.3, size=count).astype(np.float32)
    truth = DampedTrajectory(x0, v0, steps, dt, device, requires_grad=False)
    truth.rollout(
        wp.array(np.log(omega), dtype=wp.float32, device=device),
        wp.array(np.log(zeta), dtype=wp.float32, device=device),
    )
    clean = truth.positions()
    observed = clean + rng.normal(0.0, noise, size=clean.shape).astype(np.float32)
    return x0, v0, omega, zeta, observed


class Identifier:
    def __init__(self, x0, v0, observed, dt, device, omega_guess=1.0, zeta_guess=0.1):
        self.device = wp.get_device(device)
        self.count = len(x0)
        self.steps = observed.shape[0]
        self.sim = DampedTrajectory(x0, v0, self.steps, dt, device)
        self.obs = [wp.array(o, dtype=wp.vec3, device=self.device) for o in observed]
        self.params = [
            wp.array(np.full(self.count, np.log(omega_guess), np.float32), dtype=wp.float32,
                     device=self.device, requires_grad=True),
            wp.array(np.full(self.count, np.log(zeta_guess), np.float32), dtype=wp.float32,
                     device=self.device, requires_grad=True),
        ]
        self.loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        # Particles are independent, so the summed loss gives each particle its own gradient.
        self.scale = 1.0 / self.steps

    def forward(self):
        self.loss.zero_()
        xs = self.sim.rollout(*self.params)
        for t in range(self.steps):
            wp.launch(trajectory_loss, dim=self.count,
                      inputs=[xs[t + 1], self.obs[t], self.scale, self.loss], device=self.device)
        return self.loss

    def loss_and_grad(self):
        tape = wp.Tape()
        with tape:
            self.forward()
        tape.backward(loss=self.loss)
        value = float(self.loss.numpy()[0])
        grads = [p.grad.numpy().copy() for p in self.params]
        tape.zero()
        return value, grads

    def loss_value(self, log_omega, log_zeta):
        self.params[0].assign(log_omega.astype(np.float32))
        self.params[1].assign(log_zeta.astype(np.float32))
        return float(self.forward().numpy()[0])

    def fit(self, iterations=300, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
        theta = [p.numpy().astype(np.float64) for p in self.params]
        m = [np.zeros_like(t) for t in theta]
        s = [np.zeros_like(t) for t in theta]
        history = []
        for k in range(1, iterations + 1):
            value, grads = self.loss_and_grad()
            history.append(value / self.count)
            for j in range(2):
                g = grads[j].astype(np.float64)
                m[j] = beta1 * m[j] + (1 - beta1) * g
                s[j] = beta2 * s[j] + (1 - beta2) * g * g
                mhat = m[j] / (1 - beta1**k)
                shat = s[j] / (1 - beta2**k)
                theta[j] -= lr * mhat / (np.sqrt(shat) + eps)
                self.params[j].assign(theta[j].astype(np.float32))
        return np.exp(theta[0]), np.exp(theta[1]), history


def gradient_check(device="cpu", count=8, steps=60, dt=0.02, h=1e-3):
    """Compare tape gradients with central finite differences (float64 reference loss)."""
    x0, v0, _, _, observed = make_observations(count, steps, dt, noise=0.0, device=device, seed=3)
    ident = Identifier(x0, v0, observed, dt, device, omega_guess=1.2, zeta_guess=0.15)
    _, grads = ident.loss_and_grad()
    base = [p.numpy().astype(np.float64) for p in ident.params]
    worst = 0.0
    for j in range(2):
        for i in range(count):
            plus = [b.copy() for b in base]
            minus = [b.copy() for b in base]
            plus[j][i] += h
            minus[j][i] -= h
            fd = (ident.loss_value(*plus) - ident.loss_value(*minus)) / (2 * h)
            ad = float(grads[j][i])
            rel = abs(ad - fd) / max(abs(fd), 1e-6)
            worst = max(worst, rel)
    ident.loss_value(*base)
    return worst


def run_sysid(device="cpu", count=4096, steps=150, dt=0.02, noise=0.01, iterations=300):
    x0, v0, omega, zeta, observed = make_observations(count, steps, dt, noise, device)
    ident = Identifier(x0, v0, observed, dt, device)
    omega_hat, zeta_hat, history = ident.fit(iterations=iterations)
    omega_err = np.abs(omega_hat - omega) / omega
    zeta_err = np.abs(zeta_hat - zeta) / zeta
    report = {
        "device": str(device),
        "particles": count,
        "steps": steps,
        "dt_s": dt,
        "noise_std": noise,
        "iterations": iterations,
        "initial_loss": history[0],
        "final_loss": history[-1],
        "omega_median_rel_error": float(np.median(omega_err)),
        "omega_p95_rel_error": float(np.percentile(omega_err, 95)),
        "zeta_median_rel_error": float(np.median(zeta_err)),
        "zeta_p95_rel_error": float(np.percentile(zeta_err, 95)),
        "max_gradcheck_rel_error": gradient_check(device),
    }
    return report, history, omega, omega_hat, zeta, zeta_hat
