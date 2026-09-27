# Being-H

Being-H is BeingBeyond's family of human-centric embodied foundation models.
Within this repository, **Being-H0.7** is our flagship **WAM** model and **Being-H0.5** is our flagship **VLA** model.

## Model Family

| Project&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Positioning&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Summary | Links&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
|---------|-------------|---------|-------|
| [Being-H0.7](Being-H07/) | Flagship WAM | A latent world-action model from egocentric videos with future-aware latent reasoning. | [Blog](https://research.beingbeyond.com/being-h07) / [Paper](https://arxiv.org/abs/2605.00078) |
| [Being-H0.5](Being-H05/) | Flagship VLA | A human-centric VLA model for cross-embodiment generalization with a unified action space. | [Blog](https://research.beingbeyond.com/being-h05) / [Paper](https://arxiv.org/abs/2601.12993) / [Models](https://huggingface.co/collections/BeingBeyond/being-h05) |
| [Being-H0](https://github.com/BeingBeyond/Being-H/tree/being-h0) | Previous VLA | The first Being-H release for human-video VLA pretraining. | [Blog](https://research.beingbeyond.com/being-h0) / [Paper](https://arxiv.org/abs/2507.15597) / [Models](https://huggingface.co/collections/BeingBeyond/being-h0) |

## News

- **[2026-09-24]**: **Being-H0.7** is accepted by NeurIPS 2026! 🔥🔥
- **[2026-09-05]**: **Being-H0.5** is accepted by CoRL 2026! 🔥🔥
- **[2026-06-09]**: We add [Being-H-EDU](tutorials/Being-H-EDU/), an educational tutorial workspace for post-training and deployment examples.
- **[2026-05-01]**: **Being-H0** is accepted by ICML 2026! Welcome to connect with the BeingBeyond Team at the venue then! 🔥🔥 
- **[2026-04-14]**: We publish **Being-H0.7**, our flagship WAM model. See the [blog](https://research.beingbeyond.com/being-h07) and [paper](https://research.beingbeyond.com/projects/being-h07/being-h07.pdf). Code and checkpoints are coming soon!
- **[2026-03-20]**: We release the [UniHand_Preview](https://huggingface.co/datasets/BeingBeyond/UniHand_Preview) dataset, a subset of the Being-H0.5 pre-training mixture.
- **[2026-01-24]**: We update the H0.5 training, inference, and data preparation docs, and open-source post-training data for PND Adam-U through our [Hugging Face dataset collection](https://huggingface.co/collections/BeingBeyond/pnd-adam-u-data).
- **[2026-01-20]**: We publish **Being-H0.5**, our flagship VLA model for cross-embodiment generalization.
- **[2025-08-02]**: We release the **Being-H0** codebase and pretrained models through the [BeingBeyond Hugging Face collections](https://huggingface.co/collections/BeingBeyond/being-h0).
- **[2025-07-21]**: We publish **Being-H0**, our first human-video VLA release. Read the [paper](https://arxiv.org/pdf/2507.15597).


## Projects Based on Being-H

We are seeing a growing set of excellent projects built on top of the Being-H family:

- TTP: Human-Centric Transferable Tactile Pre-Training for Dexterous Robotic Manipulation [arXiv 26'04](https://arxiv.org/abs/2607.01067) | [website](https://research.beingbeyond.com/ttp) | [GitHub](https://github.com/BeingBeyond/TTP)
- Unmasking the Illusion of Embodied Reasoning in Vision-Language-Action Models. [arXiv 26'04](https://arxiv.org/abs/2604.18000) | [website](https://research.beingbeyond.com/better) | [GitHub](https://github.com/BeingBeyond/BeTTER)
- Conservative Offline Robot Policy Learning via Posterior-Transition Reweighting. [arXiv 26'03](https://arxiv.org/abs/2603.16542) | [website](https://research.beingbeyond.com/ptr) | [GitHub](https://github.com/BeingBeyond/PTR)
- DexHiL: A Human-in-the-Loop Framework for Vision-Language-Action Model Post-Training in Dexterous Manipulation. [arXiv 26'03](https://arxiv.org/abs/2603.09121) | [website](https://chenzhongxi-sjtu.github.io/dexhil/)
- Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild. [arXiv 26'02](https://arxiv.org/abs/2602.21736) | [website](https://research.beingbeyond.com/jala) | [GitHub](https://github.com/BeingBeyond/JALA)
- Rethinking Visual-Language-Action Model Scaling: Alignment, Mixture, and Regularization. [arXiv 26'02](https://arxiv.org/pdf/2602.09722) | [website](https://research.beingbeyond.com/rethink_vla) | [GitHub](https://github.com/BeingBeyond/Rethink_VLA)
- Spatial-Aware VLA Pretraining through Visual-Physical Alignment from Human Videos. [arXiv 25'12](https://arxiv.org/pdf/2512.13080) | [website](https://research.beingbeyond.com/vipa-vla) | [GitHub](https://github.com/BeingBeyond/VIPA-VLA)

Feel free to open a pull request if you want to share work built on Being-H.

## Tutorials

- [Tutorials](tutorials/) collect practical examples for adapting, post-training, evaluating, and deploying Being-H models on educational robots and community benchmarks.
- [Being-H-EDU](tutorials/Being-H-EDU/): an educational tutorial workspace for data processing, post-training, and local robot deployment with Being-H0.5. The current public example supports SO101.

## Citation

If you find the Being-H family useful, please consider citing the relevant release:

**Being-H0.7**

```bibtex
@article{beingbeyond2026beingh07,
  title={Being-H0. 7: A Latent World-Action Model from Egocentric Videos},
  author={Luo, Hao and Zhang, Wanpeng and Feng, Yicheng and Zheng, Sipeng and Xu, Haiweng and Xu, Chaoyi and Xi, Ziheng and Fu, Yuhui and Lu, Zongqing},
  journal={arXiv preprint arXiv:2605.00078},
  year={2026}
}
```

**Being-H0.5**

```bibtex
@article{beingbeyond2026beingh05,
  title={Being-H0. 5: Scaling Human-Centric Robot Learning for Cross-Embodiment Generalization},
  author={Luo, Hao and Wang, Ye and Zhang, Wanpeng and Zheng, Sipeng and Xi, Ziheng and Xu, Chaoyi and Xu, Haiweng and Yuan, Haoqi and Zhang, Chi and Wang, Yiqing and others},
  journal={arXiv preprint arXiv:2601.12993},
  year={2026}
}
```

**Being-H0**

```bibtex
@inproceedings{beingbeyond2025beingh0,
  title={Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos},
  author={Luo, Hao and Feng, Yicheng and Zhang, Wanpeng and Zheng, Sipeng and Wang, Ye and Yuan, Haoqi and Liu, Jiazheng and Xu, Chaoyi and Jin, Qin and Lu, Zongqing},
  booktitle={International Conference on Machine Learning},
  year={2026},
  organization={PMLR}
}
```

## License

This repository is released under Apache-2.0. See [LICENSE](LICENSE).
