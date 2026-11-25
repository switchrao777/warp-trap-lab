"""Time the same resident particle workload on each requested Warp device."""

import statistics
import time

import numpy as np
import warp as wp

from .simulation import METHODS, Particles, initial_state


def benchmark(devices, sizes, steps, repeats, dt=0.01):
    rows = []
    cpu_times = {}
    for count in sizes:
        x, v, omega = initial_state(count)
        for device in devices:
            for method in METHODS:
                warmup = Particles(x, v, omega, device)
                for _ in range(5):
                    warmup.step(method, dt)
                wp.synchronize_device(device)
                del warmup

                samples = []
                for _ in range(repeats):
                    particles = Particles(x, v, omega, device)
                    wp.synchronize_device(device)
                    start = time.perf_counter()
                    for _ in range(steps):
                        particles.step(method, dt)
                    wp.synchronize_device(device)
                    samples.append(time.perf_counter() - start)

                    final_x, final_v = particles.snapshot()
                    if not (np.isfinite(final_x).all() and np.isfinite(final_v).all()):
                        raise RuntimeError(f"Non-finite benchmark state on {device}")

                median = statistics.median(samples)
                if device == "cpu":
                    cpu_times[count, method] = median
                cpu_time = cpu_times.get((count, method))
                rows.append({
                    "device": str(device),
                    "device_name": wp.get_device(device).name,
                    "particles": count,
                    "integrator": method,
                    "steps": steps,
                    "dt_s": dt,
                    "repeats": repeats,
                    "warmup_steps": 5,
                    "median_ms": 1000 * median,
                    "particle_steps_per_s": count * steps / median,
                    "speedup_vs_cpu": cpu_time / median if cpu_time else None,
                    "samples_ms": [1000 * sample for sample in samples],
                })
    return rows
