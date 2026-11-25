"""Command-line experiment runner."""

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import warp as wp


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description="Run Warp harmonic-trap experiments.")
    commands = parser.add_subparsers(dest="command", required=True)
    validation = commands.add_parser("validate", help="Check the analytical and discrete predictions")
    validation.add_argument("--device", default="cpu", help="A Warp device such as cpu or cuda:0")
    perf = commands.add_parser("benchmark", help="Time both integrators over multiple batch sizes")
    perf.add_argument("--device", default="all", help="cpu, cuda:0, or all (CPU and first CUDA GPU)")
    perf.add_argument("--sizes", nargs="+", type=positive_int, default=[1024, 32768, 262144])
    perf.add_argument("--steps", type=positive_int, default=100)
    perf.add_argument("--repeats", type=positive_int, default=5)
    ident = commands.add_parser("sysid", help="Recover stiffness and damping by differentiating through the simulator")
    ident.add_argument("--device", default="cpu", help="A Warp device such as cpu or cuda:0")
    ident.add_argument("--particles", type=positive_int, default=4096)
    ident.add_argument("--steps", type=positive_int, default=150)
    ident.add_argument("--iterations", type=positive_int, default=300)
    ident.add_argument("--noise", type=float, default=0.01, help="Std. dev. of position noise")
    for command in (validation, perf, ident):
        command.add_argument("--out", type=Path, default=Path("runs"), help="Output directory")
    args = parser.parse_args()

    wp.init()
    cuda_devices = wp.get_cuda_devices()
    if args.device.startswith("cuda") and not cuda_devices:
        parser.error("a CUDA device was requested, but Warp detected no CUDA devices")
    try:
        if args.device != "all" or args.command != "benchmark":
            wp.get_device(args.device)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))

    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "warp_version": wp.__version__,
        "numpy_version": np.__version__,
        "detected_cuda_devices": [device.name for device in cuda_devices],
    }
    if args.command == "validate":
        from .plotting import plot_validation
        from .validation import validate

        report, trace, sweep = validate(args.device)
        report["environment"] = metadata
        write_csv(args.out / "energy.csv", trace)
        write_csv(args.out / "stability.csv", sweep)
        plot_validation(trace, sweep, args.out / "validation.png")
        (args.out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        for name, passed in report["checks"].items():
            print(f"{'PASS' if passed else 'FAIL'}  {name}")
        return 0 if all(report["checks"].values()) else 1

    if args.command == "sysid":
        from .plotting import plot_sysid
        from .sysid import run_sysid

        report, history, omega, omega_hat, zeta, zeta_hat = run_sysid(
            args.device, args.particles, args.steps, noise=args.noise, iterations=args.iterations)
        report["environment"] = metadata
        checks = {
            "gradients_match_finite_differences": report["max_gradcheck_rel_error"] < 2e-2,
            "loss_reaches_noise_floor": report["final_loss"] < 1.5 * 3 * args.noise**2,
            "omega_recovered": report["omega_median_rel_error"] < 1e-2,
            "zeta_recovered": report["zeta_median_rel_error"] < 5e-2,
        }
        report["checks"] = checks
        plot_sysid(history, omega, omega_hat, zeta, zeta_hat, args.out / "sysid.png")
        (args.out / "sysid.json").write_text(json.dumps(report, indent=2) + "\n")
        for name, passed in checks.items():
            print(f"{'PASS' if passed else 'FAIL'}  {name}")
        return 0 if all(checks.values()) else 1

    from .benchmark import benchmark
    from .plotting import plot_benchmark

    devices = [args.device]
    if args.device == "all":
        devices = ["cpu", str(cuda_devices[0])] if cuda_devices else ["cpu"]
        if not cuda_devices:
            print("Warp detected no CUDA device. Recording CPU results only.")
    rows = benchmark(devices, args.sizes, args.steps, args.repeats)
    payload = {"environment": metadata, "measurements": rows}
    write_csv(args.out / "benchmark.csv", rows)
    plot_benchmark(rows, args.out / "benchmark.png")
    (args.out / "benchmark.json").write_text(json.dumps(payload, indent=2) + "\n")
    for row in rows:
        speedup = row["speedup_vs_cpu"]
        comparison = f"  {speedup:.2f}× vs CPU" if row["device"] != "cpu" and speedup else ""
        print(f"{row['device']:>6}  {row['particles']:>8,}  {row['integrator']:>13}"
              f"  {row['median_ms']:>8.3f} ms  {row['particle_steps_per_s']:>12,.0f} particle steps/s"
              f"{comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
