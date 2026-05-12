# Deep-Split-Video: AI-Powered Bandwidth Optimization via Split Computing

![Project Status](https://img.shields.io/badge/Status-In--Progress-orange)
![University](https://img.shields.io/badge/Affiliation-Cukurova%20University-blue)
![Hardware](https://img.shields.io/badge/Optimized-Apple%20M1-silver)

## 🚀 Overview
This project introduces an end-to-end hybrid deep learning system designed to overcome bandwidth constraints in high-resolution video transmission. Based on the **Split Computing** paradigm, the architecture offloads heavy computational tasks to the server (Encoder) while utilizing the client’s local hardware (Apple M1 GPU) for real-time spatial and temporal enhancements (Decoder).

## 🛠️ Technical Key Features

### 1. Spatial Enhancement: Hybrid ESPCN
* **Luma-Focused Optimization**: To minimize hardware load, the model focuses exclusively on the Luminance (Y) channel, while Chrominance (Cb, Cr) channels are scaled via traditional interpolation.
* **Performance**: Achieves real-time synthesis at **~80 FPS** on Apple M1 hardware for 1080p outputs.
* **Sub-Pixel Convolution**: Uses Pixel-Shuffle layers to reconstruct high-resolution details directly from the low-resolution space.

### 2. Temporal Enhancement: MicroVFINetPro
* **Patch-Based Architecture**: A lightweight Video Frame Interpolation (VFI) module that analyzes motion-heavy areas to increase FPS without overloading the client.
* **Optical Flow Awareness**: Integrates **CoordConv** for spatial awareness and **Sobel Edge Loss** to maintain structural integrity during macro-motions.

### 3. Autonomous Communication: Asymmetric Bottleneck
* **Self-Checking Mechanism**: An autonomous "Image Complexity Inspector" that dynamically adjusts the penalty coefficient to prevent "Reward Hacking".
* **AI-Driven Compression**: Achieves up to **27.6% net bandwidth savings** by utilizing STE-based 8-bit quantization compatible with ZLIB lossless compression.

## 🔬 R&D Challenges & Engineering Lessons
This repository documents not just successes, but the critical analysis of scientific limits encountered during development:
* **The Size Paradox**: Discovered that without spatial downsampling, mapping 3-channel RGB to an 8-channel latent space increases physical data volume, leading to the **V19 vision** (mandatory downsampling-upsampling).
* **Capacity Paradox**: Addressed "Gradient Death" in shallow networks through **Curriculum Learning** (Painter vs. Sculptor phases).

## ⚠️ Current Status: Work in Progress
The project is currently in an active research phase. The following updates are expected soon:
* [ ] **Full Bibliography**: Comprehensive list of academic references and literature review.
* [ ] **V19 Model Implementation**: Integration of mandatory spatial downsampling for improved compression ratios.
* [ ] **Full Dataset & Weights**: Release of pre-trained weights and curated training pipelines.
* [ ] **Final Thesis Report**: The complete, finalized version of the graduation report.

## 🎓 Academic Affiliation
Developed as a Graduation Project at **Çukurova University**, Department of Computer Engineering.
**Advisor:** Assoc. Prof. Dr. Mehmet SARIGÜL.
**Developer:** Halit ŞEN.
