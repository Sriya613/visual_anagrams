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
  <img src="figures/System Architecture/Simple Architecture Diagram.jpg" width="800"/>
</p>

---


## Step-by-Step Working Overview (Pipeline)

The system follows a **three-stage coarse-to-fine diffusion pipeline** to generate
a single image that reveals **different meanings under specific transformations**.

### Step 1: Provide Inputs
- **Reference Image**: An image that anchors the overall structure (e.g., a face,
  object, painting, or scene). Simpler, single-subject images work best.
- **Text Prompts**: One prompt per view, describing what the image should look like
  when seen under a specific transformation (e.g., “a tiger” vs. “a human face”).
- **Transformations**: Choose invertible views such as flips, patch-based,
  triangle-based, or colour-space transformations.

---

### Step 2: Stage 1 – Low-Resolution Image-to-Image Diffusion (64×64)
- The model generates a coarse visual anagram by jointly denoising the reference
  image across all specified views.
- This stage ensures the **core semantic alignment** between prompts and
  transformations.

---

### Step 3: Stage 2 – Multi-View Super-Resolution (256×256)
- The coarse output is upsampled while reapplying the same transformations.
- Improves structural clarity and texture without breaking the illusion.

---

### Step 4: Stage 3 – High-Resolution Refinement (1024×1024)
- The image is refined into a high-resolution, photorealistic output.
- No additional noise is added, preserving multi-view consistency.

---

### Step 5: Outputs
- A **single high-resolution image** that supports multiple semantic interpretations
  under different transformations.
- Optional animations and quantitative metrics can be generated for analysis.

---

### Detailed Architecture 
The detailed architecture highlights the internal denoising loop, multi-view projections, inverse mappings, and aggregation strategy used to maintain consistency across transformations during diffusion. 
<p align="center"> <img src="figures/System Architecture/Detailed Architecture Diagram.jpg" width="900"/> </p>

---
