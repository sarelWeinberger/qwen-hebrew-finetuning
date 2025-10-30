# Cross-Language MoE Analysis for Qwen3-30B

This directory contains comprehensive analysis tools for studying Mixture of Experts (MoE) routing patterns in the Qwen/Qwen3-30B-A3B-Instruct-2507 model when processing Hebrew vs English text.

## 🔬 Research Focus

Investigating how MoE routing patterns differ between Hebrew and English language processing across all 48 transformer layers, with 128 experts per layer.

## 📁 Files Overview

### Core Analysis Scripts

- **`all_layers_moe_analysis.py`** - Complete analysis across all 48 MoE layers
- **`sample_moe_analysis.py`** - Efficient analysis sampling key layers [0,5,10,15,20,25,30,35,40,45,47]
- **`comprehensive_moe_analysis.py`** - Multi-category analysis with diverse prompt types
- **`moe_analysis.py`** - Basic MoE routing analysis framework
- **`moe_analysis_log_only.py`** - Log-only version for batch processing

### Deployment & Utils

- **`deploy.py`** - Model deployment and basic inference testing
- **`requirements.txt`** - Python dependencies (PyTorch 2.8.0, transformers, accelerate, jinja2)

### Analysis Results

- **`moe_all_layers_analysis_20250910_185847.log`** - Complete 48-layer analysis results
- **`moe_sample_analysis_20250910_182622.log`** - Sampled layer analysis results
- **`analysis_summary.md`** - Summary of key findings

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Layer Analysis
```bash
python3 all_layers_moe_analysis.py
```

### 3. Run Efficient Sampling Analysis
```bash
python3 sample_moe_analysis.py
```

## 📊 Key Findings

### Layer-by-Layer Specialization
- **Early Layers (0-10)**: High entropy (~4.0-4.2), broad expert usage
- **Middle Layers (11-30)**: Progressive specialization begins
- **Final Layers (37-47)**: Strong language differentiation

### Expert Usage Patterns
- **Hebrew**: Tends toward lower entropy in final layers (more focused routing)
- **English**: Maintains higher entropy (more distributed routing)
- **Expert Overlap**: Decreases from ~60% (early) to ~20% (final layers)

### Critical Layer 47 Differences
- **English**: Entropy = 3.875, distributed routing
- **Hebrew**: Entropy = 3.391, highly focused routing
- **Overlap**: Only 20% shared experts

## 🔧 Analysis Metrics

Each analysis calculates:

### Routing Diversity
- **Entropy**: Measures routing distribution across experts
- **Variance**: Load balancing across expert usage
- **Gini Coefficient**: Expert concentration measure

### Cross-Language Comparison
- **Expert Overlap**: Percentage of shared active experts
- **Routing Divergence**: Statistical differences in expert selection
- **Layer Progression**: How patterns evolve through network depth

## 📈 Statistical Significance

- **28.91% average expert overlap** between Hebrew and English
- **Progressive routing divergence** from 60% (early) to 20% (final layer)
- **Hebrew shows 12% lower entropy** in final layers (more specialized)

## 🎯 Use Cases

### Research Applications
1. **Multilingual Model Analysis**: Understanding language-specific neural pathways
2. **Model Optimization**: Identifying redundant vs. specialized experts
3. **Cross-Language Transfer**: Studying shared vs. distinct processing patterns

### Practical Applications
1. **Model Pruning**: Remove unused experts for efficiency
2. **Language Detection**: Use routing patterns for language identification
3. **Fine-tuning Strategy**: Target specific experts for language enhancement

## 🔬 Technical Details

### Model Architecture
- **Model**: Qwen/Qwen3-30B-A3B-Instruct-2507
- **Layers**: 48 transformer layers with MoE
- **Experts**: 128 experts per MoE layer
- **Router**: Top-2 expert selection with logits analysis

### Analysis Framework
- **Device**: CUDA-enabled GPU with device_map="auto"
- **Precision**: Auto-detection (likely BF16/FP16)
- **Batch Processing**: Single prompt analysis for precision
- **Logging**: Comprehensive timestamped analysis logs

## 📝 Citation

If you use this analysis framework in your research, please cite:

```bibtex
@misc{qwen_hebrew_moe_analysis_2025,
  title={Cross-Language MoE Routing Analysis for Hebrew-English Processing},
  author={Sarel Weinberger},
  year={2025},
  url={https://github.com/sarelWeinberger/qwen-hebrew-finetuning}
}
```

## 📞 Contact

For questions about the analysis methodology or results, please open an issue in the main repository.

---

**Last Updated**: September 10, 2025
**Analysis Date**: September 10, 2025, 18:58:47 UTC
