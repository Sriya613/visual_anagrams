# Seeing Double: Image-to-Image Diffusion for Multi-Transformation Visual Anagrams

## Introduction

Visual anagrams are multi-view optical illusions where a single image reveals
**different semantic interpretations when observed under specific transformations**
such as flips, rotations, or structured permutations. Recent diffusion-based
approaches have shown that generative models can internalize perceptual biases,
making them suitable for synthesizing such illusions.

This project presents an **image-to-image extension of the Visual Anagrams
framework**, enabling the generation of **high-resolution, multi-view visual
illusions** conditioned jointly on a **reference image and natural language prompts**.
By anchoring generation to an input image and using a **staged diffusion pipeline**,
the system preserves spatial fidelity while enforcing semantic divergence across
multiple transformed views.

In addition, this work expands the transformation space beyond standard geometric
operations by introducing **novel invertible patch-wise, triangular, and color-space
transformations**, as well as a mechanism to **compose multiple compatible
transformations within a single diffusion process**.

---

## System Architecture

### Simple Architecture Overview

The following diagram illustrates the high-level flow of the system, showing how
a reference image and text prompts are processed through a three-stage diffusion
pipeline to produce multi-view visual anagrams.

<p align="center">
  <img src="visual_anagrams/figures/System Architecture/Simple Architecture Diagram.jpg" width="800"/>
</p>

---
