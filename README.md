# REBL: Enhanced Transformer-Based Chemical-Disease Relation Extraction

## Overview

REBL (Relation Extraction for Biomedical Literature) is an enhanced transformer-based approach for extracting chemical-induced disease (CID) relations from biomedical literature. This project addresses key limitations of existing models, including standard BioBERT, through the integration of external knowledge bases, document-level processing features, and targeted recall optimization.

## Key Features

- **Enhanced BioBERT Model**: Extends BioBERT with knowledge integration and document-level processing
- **Knowledge Base Integration**: Incorporates structured domain knowledge from the Comparative Toxicogenomics Database (CTD)
- **Longformer Implementation**: Handles document-level processing with extended context windows (4,096 tokens)
- **Recall Optimization**: Implements focal loss, class weighting, and dynamic threshold tuning
- **Data Augmentation**: Uses entity swapping, synonym replacement, and CTD-based weak supervision
- **Comprehensive Evaluation**: Includes ablation studies and detailed error analysis

## Performance Results

Our enhanced model achieves significant improvements over baseline approaches:

| Model | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| CNN Baseline | 51.80% | 48.33% | 50.01% |
| BioBERT Baseline | 66.49% | 38.01% | 48.37% |
| Enhanced (Full) | **68.70%** | **51.20%** | **58.70%** |

**Key Improvements:**
- 10.33% absolute F1 score improvement over BioBERT baseline
- 13.19% recall improvement (from 38.01% to 51.20%)
- 39.47% reduction in total errors
- 50.13% reduction in distant relation errors
- 56.80% reduction in implicit relation errors

## Dataset

The project uses the **BioCreative V Chemical Disease Relation (CDR) corpus**, which includes:
- 1,500 PubMed abstracts
- 4,409 annotated chemicals
- 5,818 diseases
- 3,116 chemical-disease interactions
- Document-level annotations with cross-sentence relations

## Project Structure

```
REBL/
├── advancedREBL.ipynb          # Main implementation notebook
├── project.tex                 # Research paper (LaTeX)
├── REBL.pdf                    # Research paper (PDF)
├── saved_models/               # Trained model checkpoints
│   ├── biobert_model.pt
│   ├── improved_biobert_model.pt
│   ├── cnn_model.pt
│   └── *.pkl                   # Training history files
├── CDR_Data/                   # Dataset files
│   └── CDR.Corpus.v010516/
│       ├── CDR_TrainingSet.BioC.xml
│       ├── CDR_DevelopmentSet.BioC.xml
│       └── CDR_TestSet.BioC.xml
├── BC5CDR_Evaluation-0.0.3/    # Official evaluation toolkit
│   ├── bc5cdr_eval.jar
│   └── data/
└── baseline_model.pt           # Baseline model checkpoint
```

## Installation and Setup

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- CUDA-compatible GPU (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd REBL
   ```

2. **Install required packages:**
   ```bash
   pip install torch transformers pandas numpy networkx scikit-learn nltk spacy tqdm matplotlib seaborn wandb gensim
   ```

3. **Download NLTK resources:**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

4. **Download the CDR dataset:**
   - The dataset should be placed in the `CDR_Data/` directory
   - Ensure the BioC XML files are properly formatted

## Usage

### Running the Main Implementation

1. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook advancedREBL.ipynb
   ```

2. **Execute cells sequentially:**
   - The notebook is organized into sections covering data processing, model implementation, training, and evaluation
   - Each section can be run independently

### Model Training

The notebook includes implementations for:

1. **CNN Baseline Model**
   - Traditional convolutional neural network approach
   - Uses pre-trained BERT embeddings

2. **BioBERT Baseline**
   - Standard BioBERT implementation
   - Document-level processing with entity markers

3. **Enhanced BioBERT**
   - Knowledge base integration with CTD
   - Recall optimization techniques
   - Data augmentation

4. **Longformer Model**
   - Extended context window (4,096 tokens)
   - Document-level attention patterns

### Evaluation

The project includes comprehensive evaluation metrics:

- **Standard Metrics**: Precision, Recall, F1-score
- **Cross-sentence Analysis**: Performance on relations spanning multiple sentences
- **Error Analysis**: Detailed categorization of error types
- **Ablation Studies**: Contribution analysis of each enhancement

### Using Pre-trained Models

Pre-trained models are available in the `saved_models/` directory. **Note**: Large model files are compressed due to GitHub file size limits.

#### Decompressing Model Files

1. **Run the decompression script:**
   ```bash
   ./decompress_models.sh
   ```

2. **Manual decompression:**
   ```bash
   gunzip -k *.pt.gz
   gunzip -k saved_models/*.pt.gz
   ```

#### Loading Models

```python
import torch
from transformers import BertTokenizer, BertModel

# Load the enhanced BioBERT model
model = torch.load('saved_models/improved_biobert_model.pt')
tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
```

**Important**: The large BioBERT models (413MB each) are too large for GitHub even after compression. These models are not included in the repository. You can:

1. **Train the models locally** using the provided notebook (`advancedREBL.ipynb`)
2. **Download pre-trained models** from a file hosting service (links to be provided)
3. **Use Git LFS** (Large File Storage) if you have access to it

The repository includes:
- ✅ Baseline CNN model (12MB)
- ✅ Training history files
- ✅ Results and evaluation metrics
- ✅ Complete implementation notebook
- ❌ Large BioBERT models (413MB each) - excluded due to size limits

## Methodology

### Enhanced Architecture

1. **Knowledge Integration**
   - CTD chemical-disease interaction data
   - Knowledge graph embeddings
   - Feature fusion with attention mechanism

2. **Document-Level Processing**
   - Longformer with 4,096 token context window
   - Local and global attention patterns
   - Cross-sentence relation modeling

3. **Recall Optimization**
   - Focal loss for hard examples
   - Class weighting for imbalanced data
   - Dynamic threshold tuning
   - Multi-head classification

4. **Data Augmentation**
   - Entity swapping
   - Synonym replacement
   - CTD-based weak supervision

### Key Innovations

- **Domain Knowledge Integration**: First to integrate structured knowledge with BioBERT for chemical-disease relations
- **Document-Level BioBERT**: Extends BioBERT's capabilities for long-document processing
- **Targeted Recall Optimization**: Addresses precision-recall imbalance in relation extraction
- **Comprehensive Error Analysis**: Detailed categorization of error types and reduction strategies

## Research Contributions

This work makes several significant contributions to biomedical NLP:

1. **Enhanced BioBERT**: Extends BioBERT's capabilities for specialized chemical-disease relation extraction
2. **Knowledge Integration**: Demonstrates the value of structured domain knowledge in transformer models
3. **Document-Level Processing**: Addresses long-range dependency challenges in biomedical text
4. **Recall Optimization**: Provides techniques for balanced precision-recall performance
5. **Error Analysis**: Comprehensive analysis of relation extraction challenges and solutions

## Citation

If you use this work in your research, please cite:

```bibtex
@article{rebl2024,
  title={Enhanced Transformer-Based Chemical-Disease Relation Extraction Using BioBERT and Knowledge Integration for Biomedical Literature},
  author={He, Jinfeng},
  journal={Cornell University},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BioCreative V CDR task organizers for the dataset
- Comparative Toxicogenomics Database (CTD) for knowledge base
- Hugging Face Transformers library
- PyTorch development team

## Contact

For questions or issues related to this project, please open an issue on the repository or contact the maintainers.

---

**Note**: This implementation is based on research conducted for SYSEN5630 at Cornell University. The models and results are intended for research purposes and should be validated for production use.
