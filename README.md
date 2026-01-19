# Being-H05: Unified Vision-Language-Action Model for Multi-Embodiment Robot Manipulation

<div align="center">

[![Blog](https://img.shields.io/badge/Blog-Being--H05-green)](https://research.beingbeyond.com/being-h05)
[![arXiv](https://img.shields.io/badge/arXiv-2601.xxxxx-b31b1b.svg)](https://research.beingbeyond.com/projects/being-h05/being-h05.pdf)
[![Models](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/collections/BeingBeyond/being-h05)

</div>

Being-H0.5 is a foundational VLA model that scales human-centric learning with UniHand-2.0 and a unified action space to enable robust cross-embodiment robot control.

*(For our previous Being-H0 version, please visit the [being-h0](https://github.com/BeingBeyond/Being-H/tree/being-h0) branch.)*

## News

- **[2026-01-20]**: We release the **Being-H0.5** codebase! Check our [Hugging Face Model Collections](https://huggingface.co/collections/BeingBeyond/being-h05) for pretrained and post-trained models. 🔥🔥🔥
- **[2025-08-02]**: We release the **Being-H0** codebase and pretrained models! Check our [Hugging Face Model Collections](https://huggingface.co/collections/BeingBeyond/being-h0) for more details. 🔥🔥🔥
- **[2025-07-21]**: We publish **Being-H0**! Check our paper [here](https://arxiv.org/abs/2507.15597). 🌟🌟🌟

## Model Checkpoints

Download models from Hugging Face:

| Model Type | Model Name | Parameters | Description |
|------------|------------|------------|-------------|
| **VLA Pretrained** | [Being-H05-2B](https://huggingface.co/BeingBeyond/Being-H05-2B) | 2B | Base vision-language-action model |
| **VLA Specialist** | [Being-H05-2B_libero](https://huggingface.co/BeingBeyond/Being-H05-2B_libero) | 2B | Post-trained on LIBERO benchmark |
| **VLA Specialist** | [Being-H05-2B_robocasa](https://huggingface.co/BeingBeyond/Being-H05-2B_robocasa) | 2B | Post-trained on RoboCasa kitchen tasks |
| **VLA Generalist** | [Being-H05-2B_libero_robocasa](https://huggingface.co/BeingBeyond/Being-H05-2B_libero_robocasa) | 2B | Post-trained on both LIBERO and RoboCasa |


## Setup

### Clone repository

```bash
git clone https://github.com/BeingBeyond/Being-H05.git
cd Being-H05
```

### Create environment

```bash
conda create -n beingh python=3.10
conda activate beingh
```

### Install package

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Inference

### Quick Start

Use the pretrained or post-trained model for robot policy inference:

```python
from BeingH.inference.beingh_policy import BeingHPolicy

# Load model
policy = BeingHPolicy(
    model_path="/path/to/Being-H05-2B_libero",
    device="cuda:0"
)

# Get action from observation
action = policy.predict(
    images=images,
    state=robot_state,
    instruction="task prompt"
)
```

### Inference Server

Start an inference server for real-time robot control:

```bash
python BeingH/inference/service.py \
    --model_path /path/to/Being-H05-2B_libero \
    --port 8000 \
    --device cuda:0
```

### Evaluation on Benchmarks

Evaluate the model on LIBERO benchmark:

```bash
python BeingH/benchmark/libero/eval_libero.py \
    --model_path /path/to/Being-H05-2B_libero \
    --suite libero_spatial \
    --num_episodes 50
```

Evaluate the model on RoboCasa benchmark:

```bash
python BeingH/benchmark/robocasa/eval_robocasa.py \
    --model_path /path/to/Being-H05-2B_robocasa \
    --task PnPCounterToCab \
    --num_episodes 50
```

## Training

### Post-Training on Custom Data

Post-train the pretrained model on your own robot data:

```bash
torchrun --nproc_per_node=8 BeingH/train/train.py \
    --mllm_path /path/to/InternVL3_5-2B \
    --expert_path /path/to/Qwen3-0.6B \
    --resume_from /path/to/Being-H05-2B \
    --resume_model_only True \
    --dataset_config_file configs/posttrain/libero/libero_all.yaml \
    --output_dir /path/to/output \
    --max_steps 30000 \
    --save_steps 10000 \
    --learning_rate 1e-4 \
    --action_chunk_length 16
```

## TODO

The following features are planned for future implementation:

- [ ] Complete pretraining scripts and documentation
- [ ] Complete post-training scripts for all benchmarks
- [ ] Detailed training and data documentation
- [ ] Out-of-the-box real robot pretrained checkpoints
- [ ] Benchmark evaluation scripts for all supported tasks

## Contributing and Building on Being-H05

We encourage researchers and practitioners to leverage Being-H05 as a foundation for their own experiments and applications. Whether you're adapting Being-H05 to new robotic platforms, exploring novel manipulation tasks, or extending the model to new domains, our modular codebase is designed to support your innovations. We welcome contributions of all kinds - from bug fixes and documentation improvements to new features and model architectures. By building on Being-H05 together, we can advance the field of vision-language-action modeling and enable robots to perform more complex and diverse manipulation tasks. Join us in making robotic manipulation more capable, robust, and accessible to all.

## Acknowledgments

Being-H05 builds on the following excellent open-source projects:

- [InternVL](https://github.com/OpenGVLab/InternVL): Vision-Language model backbone
- [Qwen](https://github.com/QwenLM/Qwen): Language model and MoE expert
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO): Benchmark for lifelong robot learning
- [RoboCasa](https://github.com/robocasa/robocasa): Large-scale simulation benchmark for everyday tasks

We thank the authors for their contributions to the robotics and machine learning communities.

## License

Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.

SPDX-License-Identifier: Apache-2.0

## Citation

If you find our work useful, please consider citing us and give a star to our repository! 🌟🌟🌟

**Being-H05**

```bibtex
@misc{beingbeyond2026beingh05,
  title={Being-H0.5: Scaling Human-Centric Robot Learning for Cross-Embodiment Generalization},
  author={BeingBeyond Team},
  year={2026}
}
```
