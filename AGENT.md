Create a Python project for benchmarking GPU inference contention using RAPIDS cuML and NVML.

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