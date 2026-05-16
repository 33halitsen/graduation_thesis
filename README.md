# Deep-Split-Video: AI-Powered Bandwidth Optimization via Split Computing

## 🚀 Overview
[cite_start]This repository contains an end-to-end hybrid deep learning system designed to overcome bandwidth constraints in high-resolution video transmission based on the "Split Computing" paradigm[cite: 52]. [cite_start]By logically distributing the computational load, the system offloads heavy feature extraction to the server (Encoder) while utilizing the client’s local hardware (such as the Apple M1 GPU) for real-time spatial and temporal enhancements (Decoder)[cite: 53, 74].

## 🛠️ Technical Key Features

### 1. Spatial Enhancement: Optimized Hybrid ESPCN
* [cite_start]**Luma-Focused Optimization:** To dramatically lighten the hardware load, the model focuses exclusively on the Luminance (Y) channel, while Chrominance (Cb, Cr) channels are scaled via traditional interpolation[cite: 54, 122, 123].
* [cite_start]**Sub-Pixel Convolution:** Converts learned deep spatial relationships directly into high-resolution format through a "Pixel Shuffle" layer, bypassing processor-intensive deconvolution operations[cite: 101, 102].
* [cite_start]**Real-World Performance:** The V5 Optimized Hybrid model achieved an average of 25.20 FPS and a structural integrity of 32.55 dB PSNR during real-world 360p to 1080p concurrent upscaling tests on Apple M1[cite: 445, 464].

### 2. Temporal Enhancement: MicroVFINetPro
* [cite_start]**Patch-Based Architecture:** Replaces full-screen processing with an ultra-lightweight, patch-based network that analyzes only motion-concentrated areas to prevent speed bottlenecks[cite: 182, 183].
* [cite_start]**Spatial Awareness & Edge Preservation:** Integrates `CoordConv` (adding X and Y coordinate maps) for absolute pixel positioning and utilizes Sobel Edge Loss to prevent over-smoothing on moving edges[cite: 211, 212, 213].
* [cite_start]**Zero-Latency Jitter Buffer:** A multi-threaded queue data structure absorbs AI inference volatility, isolating the AI cycles from the rendering engine to lock playback at a highly stable, cinematic 25 FPS pacing[cite: 504, 528].

### 3. Autonomous Communication: Asymmetric Bottleneck
* [cite_start]**U-Net Inspired Asymmetric Design:** High-cost intermediate information transfers (skip connections) were canceled, forcing the network to physically compress essential details into a narrow bottleneck before transmission[cite: 78, 79, 268].
* [cite_start]**ZLIB-Compatible Quantization:** Tensor data passing through the bottleneck is converted into an 8-bit discrete integer format using Straight-Through Estimator (STE) algorithms, enabling full compatibility with lossless compression algorithms like ZLIB[cite: 57, 312].
* [cite_start]**Autonomous Self-Checking (V6):** An independent "Image Complexity Inspector" allows the model to determine its own penalty coefficient, dynamically masking unnecessary regions (Reward Hacking prevention)[cite: 318, 319, 321].

## 📊 Experimental Findings & Bandwidth Savings
* [cite_start]**Data Reduction:** The temporal AI engine acts as an autonomous network filter, bypassing up to 37.5% of downloaded frames without compromising perceived video fluency[cite: 547, 549].
* [cite_start]**Hybrid Compression Synergy:** When standard CODEC (H.264) is combined with the V6 Sparse Autonomous Masking model and ZLIB, it successfully keeps the structural quality above the 30 dB acceptability threshold (30.32 dB) while operating at ~29.0 FPS on edge devices[cite: 575, 586, 604, 628].

## 🎬 Real-World Test Videos

* ▶️ **[Test Video 1: 360p to 1080p Spatial Upscaling (Apple M1)](

https://github.com/user-attachments/assets/fbf37e39-af34-4d10-a465-62d3539b5649

)** - *Demonstrates the V5 Optimized Hybrid model's real-time inference and frame pacing capabilities.*
* ▶️ **[Test Video 2: Temporal Frame Generation (VFI)](

https://github.com/user-attachments/assets/30def65a-55b5-475b-a5b1-faee664674a5


)** - *Showcases the MicroVFINetPro architecture smoothly interpolating missing frames alongside Jitter Buffer stabilization.*
* ▶️ **[Test Video 3: Autonomous Masking & Bandwidth Saving](

https://github.com/user-attachments/assets/df526b88-ec10-4cc1-b814-48ade3705daf

)** - *Visual evidence of the V6 model autonomously masking pixels and reducing network traffic in real-time.*

## 🔬 R&D Challenges & Engineering Lessons
* **The Size Paradox:** Transferring a 3-channel (RGB) matrix to an 8-channel latent space without reducing aspect ratios expanded the data hardware-wise. [cite_start]This mathematically proved that masking channels alone is insufficient; physical Spatial Downsampling is mandatory (shaping the future V19 vision)[cite: 404, 405, 409, 410].
* **Capacity Paradox:** The model experienced "Vanishing Gradient" when faced with simultaneous quantization and quality penalties. [cite_start]This was resolved via Curriculum Learning, separating training into "Painter" (quality) and "Sculptor" (compression) phases[cite: 372, 374, 375].
* [cite_start]**Biological Isolation (YUV):** The V6 model's "color fading crisis" was tackled by mimicking the human eye's sensitivity, separating Luminance (VGG) from Chrominance (strict L1 penalty)[cite: 365, 367, 368].

## 🎓 Academic Affiliation
* [cite_start]**Institution:** Republic of Turkey, Cukurova University, Faculty of Engineering, Department of Computer Engineering[cite: 1].
* [cite_start]**Author:** Halit Şen[cite: 4].
* **Advisor:** Assoc. [cite_start]Prof. Dr. Mehmet Sarıgül[cite: 5].
* [cite_start]**Date:** 2026[cite: 6].


