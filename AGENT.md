Create a Python project for benchmarking GPU inference contention using RAPIDS cuML and NVML.

Goal: Build a reproducible GPU contention benchmark that reveals how inference performance scales and degrades under concurrent workloads.

Requirements:
- Python 3.10+
- Use cuML for kNN inference workloads
- Use pynvml for GPU telemetry
- Use multiprocessing to simulate concurrent workloads
- Use YAML config to control experiment parameters
- Log all results to CSV

Project structure:

project/
  main.py
  config.py
  workload.py
  monitor.py
  logger.py
  runner.py

  configs/
    virtual.yaml
    physical.yaml

  results/
  logs/

  requirements.txt
  README.md

Core functionality:

1) main.py
- Load config file from CLI argument
- Initialize logger
- Start telemetry monitor in background
- Launch concurrent workers via multiprocessing
- Aggregate results at the end

2) workload.py
- Initialize cuML kNN model
- Generate synthetic dataset using cupy or numpy
- Perform repeated predict() calls
- Measure latency per request
- Return throughput and latency stats

3) monitor.py
- Use pynvml to sample GPU metrics every 1 second:
  - utilization %
  - memory usage
- Store metrics in shared structure or log queue

4) runner.py
- Spawn N worker processes
- Each worker runs workload loop
- Collect results via multiprocessing Queue

5) logger.py
- Write structured CSV logs:
  timestamp, environment, num_workers, latency_ms, throughput_rps, gpu_util, gpu_mem

6) config.py
- Load YAML config
- Provide easy access to parameters

Constraints:
- Keep overhead minimal
- No heavy ML frameworks besides cuML
- Code must run on both virtual and physical GPUs without modification
- All environment differences must come from config files only

Output clean, readable, well-commented Python code.

# Experiment Execution Plan

This project implements a controlled GPU contention benchmark. The goal is to measure how inference performance degrades as concurrent GPU workloads increase.

The system MUST follow a strict experimental structure to ensure valid results.

---

## Core Principle

All experiments must:

- Use the SAME code paths
- Use the SAME workload logic
- Only vary configuration parameters (via YAML)

The ONLY intended variable is:
- Number of concurrent GPU workers
- Environment (virtual vs physical GPU)

---

## Execution Flow (High-Level)

1. Load configuration from YAML
2. Initialize logging system
3. Initialize GPU telemetry monitor (background process/thread)
4. Warm up GPU (optional but recommended)
5. Launch concurrent worker processes
6. Each worker runs repeated inference loops
7. Collect latency and throughput per worker
8. Continuously sample GPU metrics (utilization, memory)
9. Aggregate results at end of run
10. Write structured output logs (CSV)

---

## Worker Behavior

Each worker process must:

1. Initialize its own cuML kNN model
2. Generate or receive dataset (GPU-resident if possible)
3. Perform repeated inference calls:
   - model.predict(batch)
4. Measure:
   - per-request latency
   - total requests completed
5. Return summary metrics at completion

Workers must:
- Avoid unnecessary CPU-GPU data transfer
- Keep data on GPU where possible
- Minimize Python overhead inside loops

---

## Concurrency Model

- Use Python multiprocessing (NOT threading)
- Each worker is an independent process
- Workers simulate independent tenants using the same GPU

Concurrency levels to test:
- 1 worker (baseline)
- 2 workers
- 4 workers
- 6+ workers (stress)

---

## Timing Model

Each experiment run must include:

- Warmup phase (ignored in metrics)
- Measurement phase (timed)

Metrics should only be recorded AFTER warmup.

Example:
- warmup_seconds = 5
- duration_seconds = 60

---

## Metrics to Collect

### Per Worker:
- total_requests_completed
- total_time_seconds
- average_latency_ms
- throughput_requests_per_sec

### System-wide:
- GPU utilization (%)
- GPU memory usage (MB)
- timestamped samples (1 Hz)

---

## Logging Requirements

All logs must be written in structured CSV format.

Each row should include:

timestamp,
environment,
num_workers,
worker_id,
latency_ms,
throughput_rps,
gpu_utilization,
gpu_memory_MB

Logs must be:
- append-only
- consistent across runs
- written to /results directory

---

## Telemetry Monitor

The monitor must:

- Run in parallel with workload execution
- Sample GPU metrics every N seconds (default = 1)
- Use NVML via pynvml
- Record:
  - utilization.gpu
  - memory.used

The monitor must NOT:
- block workers
- introduce significant CPU overhead

---

## Experiment Loop (Critical)

The system must support running multiple experiment configurations:

Example loop:

for workers in [1, 2, 4, 8]:
    update config.num_workers
    run experiment
    save results

Each configuration must be run multiple times:

repeat_runs = 3–5

---

## Expected Behavior

As num_workers increases:

- throughput initially increases
- then plateaus (saturation point)
- latency increases sharply after saturation
- GPU utilization approaches ~100%

This produces the "knee curve" in analysis.

---

## Failure Conditions to Detect

The system should log warnings if:

- GPU utilization is low (<30%) during test
  → likely CPU bottleneck

- Throughput does not change with concurrency
  → workload not GPU-bound

- Workers crash or hang
  → resource contention too high

---

## Important Constraints

- Do NOT introduce heavy ML models for monitoring
- Monitoring must be lightweight (rule-based)
- Avoid excessive logging inside tight loops
- Ensure reproducibility across environments

---

## Latency And Efficiency Guardrails (Updated)

To avoid common implementation bugs, the runtime MUST enforce the following:

1. Duration-driven measurement is required
- `experiment.duration_seconds` must control when measurement stops.
- `workload.inference_loops` may be used as an optional cap, not the only stop condition.

2. Warmup must be excluded from metrics
- Any warmup work must finish before recording latency/throughput.

3. Worker hang protection is required
- Multiprocessing joins must use a timeout.
- Hung workers must be terminated and reported as failures.

4. Telemetry monitor must be responsive and low overhead
- Sampling loop should use event-based waiting (not blocking sleep only).
- Stopping the monitor should not add avoidable per-run delay.

5. Concurrency sweeps must be configurable
- Support a list such as `concurrency.worker_levels: [1, 2, 4, 8]`.
- Run the same workload and code path across levels.

6. Runtime diagnostics should flag invalid experiments
- Warn when GPU utilization is consistently low (<30%).
- Warn when throughput is non-positive or workers fail.

8. Hardware environment validation is mandatory
- Detect NVIDIA GPU presence and virtualization mode before running workers.
- Always log hardware detection results (including failures) to CSV.
- Block `physical` configuration unless non-virtualized NVIDIA hardware is explicitly verified.

7. Avoid hidden overhead in critical loops
- Keep CPU-side operations minimal inside inference loops.
- Keep data GPU-resident and avoid unnecessary transfers/synchronization outside measurement needs.

---

## Output Expectations

At the end of all runs, the system should produce:

- Multiple CSV files (one per run or scenario)
- Data suitable for:
  - throughput vs concurrency plots
  - latency distributions
  - utilization vs throughput graphs

---

## Goal of Implementation

The system is not just running workloads.

It must enable analysis of:

- GPU contention behavior
- performance scaling limits
- differences between virtual and physical GPUs