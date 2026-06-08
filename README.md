# Task-Profiling

## Overview

Task-Profiling is an intelligent cloud workload analysis and optimization framework designed to profile computational tasks, detect anomalies, optimize resource allocation, and monitor execution performance in virtualized cloud environments.

The framework combines deterministic feature engineering, workload characterization, anomaly detection, placement optimization, and real-time monitoring into a unified pipeline for cloud resource management.

---

## Key Features

### Task Profiling and Preprocessing
- Data normalization using Min-Max Scaling
- Noise and outlier filtering using Modified Z-Score
- Deterministic dimensionality reduction using Kaczmarz Sampling
- Resource-aware task stratification
- Workload categorization:
  - CPU-Bound Tasks
  - IO-Bound Tasks
  - Memory-Intensive Tasks
  - Balanced Tasks

### WeiDiD Feature Extraction Engine
**WeiDiD (Weibull Duration and Dependency Analyzer)** extracts:

- Temporal workload characteristics
- Delay likelihood estimation
- DAG depth estimation
- Node criticality analysis
- Fan-In and Fan-Out analysis
- Critical path detection
- Host affinity prediction
- Future load correlation analysis

### ALyFaO Priority Inversion Detection
Detects:

- Priority inversion risks
- Resource contention patterns
- Scheduling anomalies

### FeRoH Resource Envy Detection
Identifies:

- Resource envy cycles
- Fairness violations
- Resource imbalance conditions

### Falcon Task Prioritization
Provides:

- Dynamic task prioritization
- Delay-aware scheduling scores
- Multi-factor priority assessment

### WhiSOT Placement Optimization
Includes:

- Affinity bias detection
- Tabu-search-based host selection
- White Shark Optimization inspired VM placement

### Real-Time Monitoring and Control
Monitors:

- CPU overload conditions
- Memory overload conditions
- Network congestion
- Context-switch penalties
- VM health status

---

## Framework Architecture

```text
Layer 1 : Data Preprocessing
        |
        V
Layer 2 : WeiDiD Feature Extraction
        |
        V
Layer 3 : Anomaly Detection
          ├── ALyFaO
          └── FeRoH
        |
        V
Layer 4 : Placement Optimization
          └── WhiSOT
        |
        V
Layer 5 : Monitoring & Control
