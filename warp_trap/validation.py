"""Compare Warp trajectories with closed-form and discrete predictions."""

import numpy as np

from .simulation import (
    METHODS,
    POSITION,
    VELOCITY,
    Particles,
    analytical_state,
    energy,
    modified_energy,
)


def state_error(x, v, exact_x, exact_v, omega):
    """Euclidean error in phase space (x, v/omega)."""
    return float(np.sqrt(np.sum((x - exact_x)**2 + ((v - exact_v) / omega)**2)))


def validate(device):
    frequency = 2.0
    omega = np.array([frequency], dtype=np.float32)
    x0, v0 = POSITION[None, :], VELOCITY[None, :]
    dt, steps = 0.025, 800
    e0 = energy(x0, v0, omega).item()
    h0 = modified_energy(x0, v0, omega, dt).item()
    trajectories = {method: Particles(x0, v0, omega, device) for method in METHODS}
    rows = []
    for step in range(steps + 1):
        t = step * dt
        exact_x, exact_v = analytical_state(x0, v0, omega, t)
        row = {
            "step": step,
            "time_s": t,
            "exact_energy_ratio": energy(exact_x, exact_v, omega).item() / e0,
            "explicit_predicted_energy_ratio": (1 + (dt * frequency)**2)**step,
        }
        for method, particles in trajectories.items():
            x, v = particles.snapshot()
            key = method.replace("-", "_")
            row[f"{key}_energy_ratio"] = energy(x, v, omega).item() / e0
            row[f"{key}_state_error"] = state_error(x, v, exact_x, exact_v, frequency)
            if method == "semi-implicit":
                row["modified_energy_ratio"] = modified_energy(x, v, omega, dt).item() / h0
            if step < steps:
                particles.step(method, dt)
        rows.append(row)

    rates = {}
    for method in METHODS:
        errors = []
        for count in (80, 160, 320, 640):
            h = 4.0 / count
            particles = Particles(x0, v0, omega, device)
            for _ in range(count):
                particles.step(method, h)
            x, v = particles.snapshot()
            exact_x, exact_v = analytical_state(x0, v0, omega, 4.0)
            errors.append(state_error(x, v, exact_x, exact_v, frequency))
        rates[method] = {
            "dt_s": [4.0 / n for n in (80, 160, 320, 640)],
            "final_state_errors": errors,
            "observed_order": float(np.log2(errors[-2] / errors[-1])),
        }

    # Ratios are calculated in float64 on the host from float32 Warp states.
    explicit_law = max(
        abs(r["explicit_energy_ratio"] / r["explicit_predicted_energy_ratio"] - 1)
        for r in rows
    )
    exact_drift = max(abs(r["exact_energy_ratio"] - 1) for r in rows)
    semi_drift = max(abs(r["semi_implicit_energy_ratio"] - 1) for r in rows)
    modified_drift = max(abs(r["modified_energy_ratio"] - 1) for r in rows)
    s = dt * frequency
    semi_bound = s / (1 - s / 2)
    stability = stability_sweep(device)
    at = {r["omega_dt"]: r for r in stability}
    checks = {
        "exact_energy_constant": exact_drift < 1e-12,
        "explicit_matches_discrete_energy_law": explicit_law < 1e-4,
        "semi_implicit_energy_within_analytic_bound": semi_drift <= semi_bound + 1e-4,
        "semi_implicit_modified_energy_constant": modified_drift < 1e-4,
        "both_integrators_first_order": all(
            0.9 <= result["observed_order"] <= 1.15 for result in rates.values()
        ),
        "sweep_brackets_semi_implicit_boundary": (
            at[1.9]["semi_implicit_energy_ratio"] < 30
            and at[2.0]["semi_implicit_energy_ratio"] > 100
            and at[2.1]["semi_implicit_energy_ratio"] > 1e20
        ),
    }
    summary = {
        "device": str(device),
        "dt_s": dt,
        "omega_rad_s": frequency,
        "steps": steps,
        "exact_max_abs_relative_energy_drift": exact_drift,
        "explicit_final_energy_ratio": rows[-1]["explicit_energy_ratio"],
        "explicit_max_relative_error_vs_discrete_law": explicit_law,
        "semi_implicit_max_abs_relative_energy_drift": semi_drift,
        "semi_implicit_relative_energy_bound": semi_bound,
        "modified_max_abs_relative_energy_drift": modified_drift,
        "convergence": rates,
        "sweep_steps": 64,
        "checks": checks,
    }
    return summary, rows, stability


def stability_sweep(device):
    # One batch puts every frequency through the same Warp kernel per step.
    # Using dt=1 keeps the h*omega=2 sample exactly representable in float32.
    scaled_frequencies = np.round(np.linspace(0.05, 2.15, 211), 2)
    omega = scaled_frequencies.astype(np.float32)
    x0 = np.broadcast_to(POSITION, (len(omega), 3)).copy()
    v0 = np.broadcast_to(VELOCITY, (len(omega), 3)).copy()
    e0 = energy(x0, v0, omega)
    rows = [
        {
            "omega_dt": s,
            "exact_energy_ratio": 1.0,
            "explicit_predicted_energy_ratio": (1 + s**2)**64,
        }
        for s in scaled_frequencies.tolist()
    ]
    for method in METHODS:
        particles = Particles(x0, v0, omega, device)
        for _ in range(64):
            particles.step(method, 1.0)
        x, v = particles.snapshot()
        key = f"{method.replace('-', '_')}_energy_ratio"
        ratios = (energy(x, v, omega) / e0).tolist()
        for row, ratio in zip(rows, ratios, strict=True):
            row[key] = ratio
    return rows
