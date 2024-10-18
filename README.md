# [IEEE TMM 2024] EISNet: A Multi-Modal Fusion Network for Semantic Segmentation with Events and Images

This repository is an official implementation of [EISNet](https://ieeexplore.ieee.org/document/10477577).

<div align="center">
  <img src="figs/Overview_EISNet.jpg"/>
</div><br/>

If you use any of this code, please cite the following publication:

```bibtex
@article{xie2024eisnet,
  title={{EISNet}: A Multi-Modal Fusion Network for Semantic Segmentation with Events and Images},
  author={Xie, Bochen and Deng, Yongjian and Shao, Zhanpeng and Li, Youfu},
  journal={IEEE Trans. Multimedia},
  volume={26},
  pages={8639--8650},
  year={2024}
}
```

## Overview
In this project, we propose a multi-modal fusion network (EISNet) for semantic segmentation with events and images, which is comprised of two key components: Activity-Aware Event Integration Module (AEIM) and Modality Recalibration and Fusion Module (MRFM). AEIM integrates the visual cues of event data into frame-based representations that encode rich and high-confidence information via scene activity modeling. MRFM fully considers the characteristics of two modalities to achieve adaptive feature aggregation through modality recalibration and gated cross-attention fusion. Extensive experiments and ablation studies demonstrate the effectiveness and robustness of EISNet in challenging scenarios.

## Evaluation
You can download the model weights of EISNet on the DDD17 and DSEC-Semantic datasets as follows.
|Dataset|Event Encoder|Image Encoder|Resolution|mIoU (%)|Download Link|
|:-:|:-:|:-:|:-:|:-:|:-:|
|DDD17|MiT-B0|MiT-B2|200*346|75.03|[[Google Drive](https://drive.google.com/file/d/1U7XeTev_-vYxe-7VlH-1e6yZEmDjpze9/view?usp=sharing)] / [[OneDrive](https://portland-my.sharepoint.com/:u:/g/personal/boxie4-c_my_cityu_edu_hk/ETg_ok2J_l1DuxEX8PVqmv4B5aD6PkHm9FgqO-C51yI7Rg?e=75eACg)]|
|DSEC-Semantic|MiT-B0|MiT-B2|440*640|73.07|[[Google Drive](https://drive.google.com/file/d/13B_lU_dtguXpzQZgMEiMH7VePXjam0h1/view?usp=sharing)] / [[OneDrive](https://portland-my.sharepoint.com/:u:/g/personal/boxie4-c_my_cityu_edu_hk/EaQuNlXWRk5LtfZEuqsa6ecB-CDnVewEux4Ic6AW-8D5ZA?e=0dO2Ka)]|
