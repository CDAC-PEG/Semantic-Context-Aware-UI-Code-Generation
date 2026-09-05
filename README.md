# Semantic Context-Aware Automated Front-End Code Generation

Implementation of the research framework **Semantic Context-Aware Automated Front-End Code Generation for Mobile Applications using a Vision-Language Code Transformer**.

## Overview

The framework converts mobile UI design images into front-end source code through the following pipeline:

```text
UI Design Image
  -> Foreground-Aware Preprocessing
  -> SPPYOLO-v8 UI Component Detection
  -> OCR and Attribute Extraction
  -> DSFBSCAN Similar UI Element Grouping
  -> VisualTABERT Semantic Context Modeling
  -> APKT Coordinate Alignment
  -> Structured Intermediate Representation
  -> CodeLAT5+
  -> Generated Front-End Code
```

## Proposed Components

### Foreground-Aware Preprocessing
Local image regions are analyzed using intensity mean and variance. Predominantly white or near-uniform background tiles are preserved, while foreground-containing regions are enhanced using CLAHE.

### SPPYOLO-v8
SPPYOLO-v8 detects and localizes UI components. The SPPF mechanism supports multi-scale contextual feature extraction.

### OCR and Attribute Extraction
OCR extracts UI text together with attributes such as confidence values and bounding-box coordinates.

### DSFBSCAN
DSFBSCAN groups structurally and semantically related UI elements using extracted visual, textual, and spatial attributes.

### VisualTABERT
VisualTABERT models semantic relationships among UI elements by integrating textual, visual, structural, and positional information with Top-k Attention.

### APKT
APKT aligns UI-element coordinates using affine transformation and polynomial-kernel-based modeling of nonlinear spatial relationships.

### Structured Intermediate Representation
Outputs from detection, OCR, grouping, semantic modeling, and coordinate alignment are integrated into a structured representation.

### CodeLAT5+
CodeLAT5+ generates front-end source code from the structured representation. Longformer Attention is incorporated to support long input sequences and long-range dependencies.

## Repository Structure

```text
SOURCE_CODE_GENERATION_UIDESIGN/
├── Image_Preprocessing/
├── Object_Detection/
├── SCGUI/
│   └── Clustering/
├── Semantic_Context_Aware_Extraction/
├── Coordinate_alignment/
├── Code/
├── Dataset/
├── Models/
├── Graphs/
├── Output/
├── Result/
└── Run/
```

Key implementation files include `Proposed_SPPYOLO.py`, `Proposed_DSFBSCAN.py`, `Proposed_VisualTABERT.py`, `Proposed_APKT.py`, and `Proposed_CodeLAT5Plus.py`.

## Installation

Python 3.10 is recommended.

```bash
pip install -r requirements.txt
```

Tesseract OCR must also be installed separately for the OCR stage.

## Running

Configure the input, dataset, model-weight, and output paths used by the modules, then run:

```bash
python Run/Run.py
```

Individual modules and comparative-analysis scripts can also be executed separately.

## Evaluation

The implementation supports evaluation of UI component detection, clustering, semantic context modeling, coordinate alignment, and front-end code generation.

The code-generation stage can be evaluated using Perplexity, Runtime Success Rate, CodeBLEU, Structural Accuracy, Semantic Accuracy, and rendered-interface SSIM where the corresponding experimental implementation is available.

Component-wise ablation experiments can be conducted by removing individual framework components while maintaining the same dataset partition and CodeLAT5+ experimental configuration.

## Dataset

The experiments use the mobile UI design dataset described and cited in the associated manuscript. Dataset redistribution must comply with the license and terms of the original dataset provider.

## Model Weights

Trained weights can be stored under `Models/`. Large weight files may need Git LFS or an external archival location.

## Reproducibility

For reproduction of the reported experiments, use the same dataset partitions, parameter settings, model checkpoints, and evaluation procedures described in the associated manuscript.

## Citation

If you use this repository in academic work, please cite the associated research article. Full citation details will be added after publication.

## Author

**Suji Jose**  
Department of Computer Science  
Cochin University of Science and Technology (CUSAT)

## License


