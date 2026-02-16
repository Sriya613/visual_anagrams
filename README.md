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

### What Is a Transformation?
A **transformation** is an invertible operation that produces an alternative view of
the same image (such as a flip, rotation, or patch rearrangement), enabling a
single image to reveal different meanings under different views.

Most visual anagram pipelines rely on a restricted set of well-known, predefined
noise-preserving transformations, including **Horizontal and Vertical Flips**, **180° Rotation**, **90° Rotation**, **Colour Contrast / Channel Reweighting**, **Jigsaw Transformations**, **Patch Permutation (reordering fixed patches)**.
To expand the expressive power of visual anagrams, we introduce **new classes
of fully invertible transformations** that operate at patch, geometric, and
colour-space levels. Below are qualitative examples for each.

---

### Patch Flip Transformations (`patch_flip/`)

Patch flip transformations divide the image into fixed-size patches and apply
independent **horizontal or vertical flips within each patch**, introducing
strong local variation while preserving global invertibility. The number of patches can vary
but must be a multiple of 8.

<p align="center">
  <img src="figures/Transformations' Illustrations/Patch Based Horizontal Flip.png" width="760"/>
</p>

Animation:
<p align="center">
  <img src="figures/animations/horizontal flip 2.gif" width="380"/>
  <img src="figures/animations/horizontal flip 3.gif" width="380"/>
</p>

Examples:
<p align="center">
  <img src="figures/Output Images/Horizontal Patch Flipping.png" width="800"/>
</p>

**Vertical Flip**

<p align="center">
  <img src="figures/Transformations' Illustrations/Patch Based Vertical Flip.png" width="800"/>
</p>

Animation:
<p align="center">
  <img src="figures/animations/vertical flip.gif" width="325"/>
  <img src="figures/animations/vertical flip 2.gif" width="580"/>
</p>


Examples:
<p align="center">
  <img src="figures/Output Images/Vertical Patch Flipping.png" width="800"/>
</p>

**Randomised Patch Flipping**
<p align="center">
  <img src="figures/animations/randomized patch flipping.gif" width="600"/>
</p>


---

### Triangle-Based Transformations (`triangle_views/`)

Triangle-based views partition the image into fixed triangular regions and apply
structured permutations or 180° flips combined with triangle swaps. These
transformations preserve grid alignment while enabling stronger geometric
divergence than standard patch permutations.

**Triangle Patch Permutation**

<p align="center">
  <img src="figures/Transformations' Illustrations/Triangle Patch Permute.png" width="800"/>
</p>

Animation:
<p align="center">
  <img src="figures/animations/triangle permute.gif" width="380"/>
  <img src="figures/animations/triangle permute 2.gif" width="380"/>
</p>



**Triangle Patch Flipping**

<p align="center">
  <img src="figures/Transformations' Illustrations/Triangle Patch Flip.png" width="800"/>
</p>

Animation:
<p align="center">
  <img src="figures/animations/triangle flip.gif" width="380"/>
  <img src="figures/animations/triangle flip 2.gif" width="380"/>
</p>

Examples:
<p align="center">
  <img src="figures/Output Images/Triangle Flip and Permute.png" width="800"/>
</p>

📽️ *Animations demonstrating triangle permutations and flips are provided in*  
`figures/animation/`

---

### Colour Permutation Transformations (`colour_permute/`)

Colour permutation views operate directly in RGB colour space using **orthonormal
transformations**, enabling semantic divergence through colour changes without
altering spatial structure.

<p align="center">
  <img src="figures/Transformations' Illustrations/Color Permute.png" width="800"/>
</p>

Animations:
<p align="center">
  <img src="figures/animations/color contrast 1.gif" width="380"/>
  <img src="figures/animations/color contrast 2.gif" width="380"/>
</p>

Examples:
<p align="center">
  <img src="figures/Output Images/Color Permute and Rotate.png" width="800"/>
</p>

📽️ *Colour-space animation examples can be found in*  
`visual_anagrams/figures/animation/`

---

## Combined Transformations

Combined transformations allow **multiple compatible transformations** to be
applied **simultaneously within a single diffusion process**, enabling more
expressive and complex visual anagrams than single-view setups.

Instead of treating each transformation independently, compatible operations
(e.g., patch flipping + colour permutation, rotation + colour shift) are
**composed into a single view** with a well-defined inverse. This ensures that
noise predictions from different transformed views can still be mapped back to a
shared canonical space and aggregated consistently during diffusion.

<!-- <p align="center">
  <img src="figures/Transformations' Illustrations/Combined Views Transformation.png" width="850"/>
</p> -->

##Demonstration
<p>Here we provided the text prompts as "A sketch of a skull" and "A sketch of Albert Einstein" and chose thr transformations "Rotate by 180 degrees" and "Colour Negation".
This is how the combination of transformations works.</p>

<p align="center">
  <img src="figures/animations/Text to Image Merged Transformation.gif" width="500"/>
</p>

---

### Image-to-Image Combined Transformations

In the **image-to-image setting**, combined transformations are applied while
anchoring generation to a reference image. This preserves global structure while
allowing multiple transformations to jointly influence the final output. As a
result, the generated image maintains spatial coherence across views while
revealing distinct semantics under different composite transformations.

<p align="center">
  <img src="figures/Output Images/Image to Image Merged Transformation.png" width="850"/>
</p>

---

### Text-to-Image Combined Transformations

In the **text-to-image setting**, combined transformations are driven purely by
text prompts corresponding to different views. The model learns to encode
multiple semantic interpretations within a single image while satisfying all
composed transformation constraints, without relying on an explicit reference
image.

<p align="center">
  <img src="figures/Output Images/Text to Image Merged Transformation.png" width="850"/>
</p>

---

### Why This Is Novel

Prior visual anagram pipelines typically support **only one transformation per
view**, limiting the complexity and expressiveness of generated illusions. Our
approach introduces a **composable multi-view formulation** that enables:

- Joint enforcement of multiple transformations in a single diffusion trajectory
- Greater perceptual ambiguity and richer multi-view interpretations
- Consistent behavior across both image-to-image and text-to-image pipelines

This significantly expands the design space of diffusion-based visual anagrams,
allowing more controlled, flexible, and visually compelling multi-view illusions.

