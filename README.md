# VIRTUALIZATION & TESTING, NVIDIA GPU UNDER ARTIFICIAL LOAD

Author: Andrew Pierre
Cloud Computing, Wentworth Institute

# Purpose / Problem Statement

## Motivating Example: The Cost of Hidden GPU Inefficiency

### Scenario: An Early-Stage AI Researcher

Maya is training a transformer-based medical imaging model on a limited research budget.
Her lab relies on four rented cloud virtual GPU (vGPU) instances that advertise identical
compute capacity. Training runs are distributed across these instances and often take days.

As experimentation scales, Maya observes four recurring problems:

- Completion time varies by as much as 30% across similar runs.
- Reported GPU utilization remains high while end-to-end training time worsens.
- Scaling from 2 to 4 vGPUs delivers little throughput gain.
- Training stalls are more frequent during peak cloud usage windows.

Because the GPUs are virtualized, the root cause is hidden. Maya cannot directly determine
whether slowdown comes from tenant co-location, memory bandwidth pressure, or uneven
hypervisor-level resource sharing. She can verify allocation, but not behavior.

This is the core systems gap: cloud schedulers allocate accelerator resources, but they often
do not provide workload-aware visibility or contention-sensitive feedback for tuning AI jobs.
In practice, vGPU resources are managed as static units, even though their performance is
dynamic and sensitive to co-tenant interference.

## Proposed Direction

This project explores an accelerator-aware observability and control layer that operates above
the hypervisor and focuses on workload behavior rather than allocation status alone.

Desired capabilities include:

- Continuous GPU telemetry collection (kernel latency, utilization, memory pressure)
- Workload classification by compute and memory behavior
- Detection of multi-tenant contention across virtual GPU slices
- Scheduling or workload adjustment recommendations to reduce interference
- Scaling-efficiency diagnostics across distributed training nodes

Expected outputs include:

- Bottleneck attribution reports
- Variance tracking over time
- Contention alerts
- Cross-instance performance comparisons

With this feedback loop, researchers like Maya can stabilize throughput, improve scaling
efficiency, and reduce cost per experiment.

## Research Focus

This work does not replace cloud infrastructure. It augments existing virtualization and
scheduling systems with low-overhead, accelerator-sensitive observability.

Research question:

Can a distributed accelerator-aware control framework reduce contention, stabilize runtime
variance, and improve training efficiency in multi-tenant AI environments?


## Project Goals / Success Crit
Success Statement:

The project will be considered successful if the accelerator-aware control framework reduces training throughput variance by at least 30% in multi-tenant virtual GPU environments while maintaining telemetry overhead below 3% CPU utilization. Additionally, the system must detect GPU contention events with ≥85% accuracy, attribute performance bottlenecks correctly in ≥80% of experiments, and maintain scaling efficiency above 70%

# GPU Contention Benchmark (cuML + NVML)

## Overview

This project benchmarks GPU inference performance under concurrent workloads to study resource contention and scaling behavior.

It compares performance across:

* Virtualized GPU environments (vGPU)
* Physical GPU hardware

## Features

* GPU-accelerated kNN inference using RAPIDS cuML
* Multiprocessing workload generation
* Real-time GPU telemetry via NVML
* Structured logging for analysis
* Configurable experiment parameters

## Project Structure

```
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
```

## Setup

### 1. Environment

Install RAPIDS (cuML) using Conda:

```
conda create -n rapids-env -c rapidsai -c nvidia -c conda-forge \
    rapids=24.02 python=3.10 cuda-version=12.0
conda activate rapids-env
```

Install Python dependencies:

```
pip install -r requirements.txt
```

### 2. Verify GPU Access

```
nvidia-smi
```

## Running Experiments

### Virtual GPU

```
python main.py --config configs/virtual.yaml
```

### Physical GPU

```
python main.py --config configs/physical.yaml
```

## Output

Results are written to:

```
results/
```

CSV files include:

* `results/summary.csv`
* `results/workers.csv`

Each run logs:

* latency (ms)
* throughput (requests/sec)
* GPU utilization (%)
* GPU memory usage (MB)

## Experiment Design

We evaluate:

* Throughput scaling vs concurrency (knee curve)
* Throughput variance across runs
* GPU utilization vs performance
* Behavior under contention

## Notes

* All experiments use identical code paths
* Only configuration differs between environments
* Repeat runs are required for statistical validity

## Future Work

* Adaptive scheduling based on telemetry
* Multi-GPU scaling
* Real-time monitoring dashboard
