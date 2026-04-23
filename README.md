# CS5588-week13: AI Animal Care & Pet Health Visualization System

## Overview

This project implements a data-driven, controlled image generation system using Stable Diffusion for veterinary care and animal welfare visualization. The system generates realistic or educational images representing various animal health scenarios based on structured input data.

## Features

- **Structured Input Processing**: Converts animal type, breed, condition, and environment data into detailed prompts
- **Controlled Generation**: Uses Stable Diffusion with optional ControlNet conditioning
- **Prompt Engineering**: Implements structured prompt templates and negative prompts
- **Evaluation Metrics**: Measures prompt alignment, consistency, diversity, and visual quality
- **Comparison Analysis**: Compares naive vs structured prompt approaches

## Project Structure

```
├── data_loader.py          # Data loading and sample generation
├── prompt_generator.py     # Prompt engineering from structured data
├── image_generator.py      # Stable Diffusion image generation
├── evaluator.py           # Image evaluation metrics
├── main.py                # Main experiment runner
├── requirements.txt       # Python dependencies
└── outputs/               # Generated images and results
```

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main experiment:
```bash
python main.py
```

Choose the output style:
```bash
$env:IMAGE_STYLE="realistic"; python main.py
```

or

```bash
$env:IMAGE_STYLE="educational"; python main.py
```

This will:
- Generate sample animal data
- Create structured prompts
- Generate images using Stable Diffusion
- Evaluate results with multiple metrics
- Compare naive vs structured approaches
- Save results to `outputs/` directory

## Control Mechanisms

The system implements several control mechanisms:

1. **Structured Prompt Templates**: Condition-specific prompt generation
2. **Negative Prompts**: Avoid unwanted elements (blurry, cartoon, etc.)
3. **Optional ControlNet**: Canny edge conditioning for better control

## Evaluation Metrics

- **Prompt Alignment**: CLIP score measuring text-image similarity
- **Consistency**: Similarity between generated variations
- **Diversity**: Variation among generated images
- **Visual Quality**: Brightness, contrast, and sharpness analysis

## Datasets

The system is designed to work with:
- Oxford-IIIT Pet Dataset
- AFHQ Dataset (Animal Faces)
- Stanford Dogs Dataset

These datasets are excellent for improving breed realism and animal appearance consistency. At the moment, this repo uses structured metadata and synthetic sample records by default; the datasets are not automatically downloaded or ingested yet.

## AI Tools Used

### Development Tools
- **GitHub Copilot**: Assisted in code generation, debugging, and documentation
  - Generated initial code structure and classes
  - Helped with PyTorch and diffusers integration
  - Suggested evaluation metrics implementation

### Libraries & Frameworks
- **Hugging Face Diffusers**: Stable Diffusion pipeline implementation
- **OpenAI CLIP**: For prompt-image alignment evaluation
- **ControlNet**: Optional conditioning for controlled generation

### Transparency Statement
All code was developed with AI assistance from GitHub Copilot. The AI helped accelerate development but all architectural decisions and final implementations were reviewed and approved by the developer. AI-generated suggestions were validated for correctness and adapted to the specific requirements of the animal care visualization system.

## Ethics & Safety

- All outputs are clearly marked as AI-generated illustrations
- Content focuses on educational veterinary scenarios
- Avoids harmful or misleading representations
- Promotes responsible AI use in animal welfare

## Results

The system demonstrates improved performance with structured prompts:
- Higher CLIP scores for prompt alignment
- Better consistency in generated variations
- Maintained diversity while improving relevance

## Future Enhancements

- Integration with real veterinary datasets
- Advanced ControlNet conditioning (pose, segmentation)
- Multi-modal evaluation with domain experts
- Web interface for interactive generation
