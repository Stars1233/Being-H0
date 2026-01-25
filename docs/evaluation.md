# Evaluation Guide

This guide covers running benchmark evaluations for Being-H on LIBERO and RoboCasa.

## LIBERO Benchmark

### Overview

LIBERO is a benchmark for lifelong robot learning with 130 tasks across 4 task suites:
- **LIBERO-Spatial**: Spatial relationship tasks (10 tasks)
- **LIBERO-Object**: Object manipulation tasks (10 tasks)
- **LIBERO-Goal**: Goal-conditioned tasks (10 tasks)
- **LIBERO-Long**: Long-horizon tasks (10 tasks)

### Environment Setup

LIBERO requires Python 3.8:

```bash
conda create -n libero python=3.8
conda activate libero
pip install libero
pip install -r requirements.txt
```

### EGL Configuration

For headless rendering:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

### Running Evaluation

Use the provided evaluation script:

```bash
# Edit configuration in the script first
vim scripts/eval/eval-libero.sh

# Run evaluation
bash scripts/eval/eval-libero.sh
```

---

## RoboCasa Benchmark

### Overview

RoboCasa is a large-scale simulation benchmark for everyday household tasks:
- 100+ kitchen tasks
- Multiple robot configurations
- Realistic kitchen environments

### Environment Setup

RoboCasa requires Python 3.10:

```bash
conda create -n robocasa python=3.10
conda activate robocasa
pip install robocasa
python -m robocasa.scripts.download_kitchen_assets
```

### Running Evaluation

Use the provided evaluation script:

```bash
# Edit configuration in the script first
vim scripts/eval/eval-robocasa.sh

# Run evaluation
bash scripts/eval/eval-robocasa.sh
```
