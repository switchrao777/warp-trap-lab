"""Render validation and benchmark figures from recorded rows."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPLICIT = "#a84318"
SEMI = "#006c68"


def plot_validation(trace, sweep, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), layout="constrained")
    fig.suptitle("Harmonic trap | Energy and numerical stability", fontsize=15)

    t = [r["time_s"] for r in trace]
    ax = axes[0]
    ax.plot(t, [r["explicit_energy_ratio"] for r in trace], color=EXPLICIT, label="Explicit Euler")
    ax.plot(t, [r["semi_implicit_energy_ratio"] for r in trace], color=SEMI, label="Semi-implicit")
    ax.plot(t, [r["exact_energy_ratio"] for r in trace], "--", color="#555555", label="Analytical")
    ax.set(xlabel="Time (s)", ylabel="Physical energy / initial energy", title="Energy over time")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(t, [100 * (r["semi_implicit_energy_ratio"] - 1) for r in trace],
            color=SEMI, label="Physical energy")
    ax.plot(t, [100 * (r["modified_energy_ratio"] - 1) for r in trace],
            "--", color="#555555", label="Modified energy")
    ax.set(xlabel="Time (s)", ylabel="Relative energy change (%)", title="Semi-implicit energy",
           ylim=(-2.7, 4.1))
    ax.legend(frameon=False, loc="upper left")

    ax = axes[2]
    s = [r["omega_dt"] for r in sweep]
    ax.semilogy(s, [r["explicit_energy_ratio"] for r in sweep], color=EXPLICIT, label="Explicit Euler")
    ax.semilogy(s, [r["semi_implicit_energy_ratio"] for r in sweep], color=SEMI, label="Semi-implicit")
    ax.axhline(1, color="#a1a1a1", linewidth=0.8)
    ax.axvline(2.0, linestyle="--", color="#555555", label="Stability boundary")
    ax.set(xlabel=r"Scaled time step  $\omega\Delta t$", ylabel="Final physical energy / initial energy",
           title="Stability sweep | 64 steps")
    ax.legend(frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.16)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_benchmark(rows, path):
    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    for device in dict.fromkeys(row["device"] for row in rows):
        for method, color in (("explicit", EXPLICIT), ("semi-implicit", SEMI)):
            series = sorted(
                (r for r in rows if r["device"] == device and r["integrator"] == method),
                key=lambda r: r["particles"],
            )
            ax.loglog(
                [r["particles"] for r in series],
                [r["particle_steps_per_s"] for r in series],
                marker="o",
                linestyle="--" if device == "cpu" else "-",
                color=color,
                label=f"{device} · {method}",
            )
    ax.set(title="Warp integration throughput", xlabel="Particles per step", ylabel="Particle steps / s")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.16)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_sysid(history, omega, omega_hat, zeta, zeta_hat, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), layout="constrained")
    fig.suptitle("Differentiable system identification | gradients through the Warp simulator",
                 fontsize=15)
    ax = axes[0]
    ax.semilogy(range(1, len(history) + 1), history, color=SEMI)
    ax.set(xlabel="Adam iteration", ylabel="Trajectory MSE per particle", title="Loss")
    for ax, true, est, name in ((axes[1], omega, omega_hat, "omega (rad/s)"),
                                (axes[2], zeta, zeta_hat, "damping ratio zeta")):
        ax.scatter(true, est, s=2, alpha=0.35, color=SEMI)
        lo, hi = float(min(true.min(), est.min())), float(max(true.max(), est.max()))
        ax.plot([lo, hi], [lo, hi], "--", color="#555555", linewidth=1)
        ax.set(xlabel=f"True {name}", ylabel=f"Recovered {name}", title=f"Recovered {name.split()[0]}")
    fig.savefig(path, dpi=160)
    plt.close(fig)
