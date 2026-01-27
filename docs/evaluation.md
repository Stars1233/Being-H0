# Evaluation Guide

This guide covers running benchmark evaluations for Being-H on LIBERO and RoboCasa.

## LIBERO Benchmark

### Overview

LIBERO is a benchmark for lifelong robot learning with 130 tasks across 4 task suites:
- **LIBERO-Spatial**: Spatial relationship tasks (10 tasks)
- **LIBERO-Object**: Object manipulation tasks (10 tasks)
- **LIBERO-Goal**: Goal-conditioned tasks (10 tasks)
- **LIBERO-Long**: Long-horizon tasks (10 tasks)

### Install LIBERO Repository

Please follow the original [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) installation instructions:

```bash
conda create -n libero python=3.8
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
# Download standard LIBERO datasets
python benchmark_scripts/download_libero_datasets.py
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

### Install RoboCasa Repository

Please follow the original [RoboCasa](https://github.com/robocasa/robocasa) installation instructions:

1. Create and activate conda environment:
```bash
conda create -c conda-forge -n robocasa python=3.10
conda activate robocasa
```

2. Install robosuite and robocasa:
```bash
git clone https://github.com/ARISE-Initiative/robosuite
cd robosuite
pip install -e .
cd ..
git clone https://github.com/robocasa/robocasa
cd robocasa
pip install -e .
pip install pre-commit; pre-commit install           # Optional: set up code formatter.

(optional: if running into issues with numba/numpy, run: conda install -c numba numba=0.56.4 -y)
```

3. Download kitchen assets and set up macros:
```bash
python robocasa/scripts/download_kitchen_assets.py   # Caution: Assets to be downloaded are around 5GB.
python robocasa/scripts/setup_macros.py              # Set up system variables.
```

### Running Evaluation

Use the provided evaluation script:

```bash
# Edit configuration in the script first
vim scripts/eval/eval-robocasa.sh

# Run evaluation
bash scripts/eval/eval-robocasa.sh
```
