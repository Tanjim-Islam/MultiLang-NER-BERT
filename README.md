# MultiLang-NER-BERT

A multilingual Named Entity Recognition (NER) system built with BERT (Bidirectional Encoder Representations from Transformers). This project leverages the `bert-base-multilingual-cased` model to perform token-level classification for identifying named entities across multiple languages.

## 🚀 Features

- **Multilingual Support**: Train and evaluate NER models on various languages
- **BERT-based Architecture**: Uses `bert-base-multilingual-cased` for robust performance
- **CoNLL Format Support**: Works with standard CoNLL formatted datasets
- **Comprehensive Evaluation**: Provides accuracy, precision, recall, and F1-score metrics
- **Configurable Training**: Easily adjustable hyperparameters and settings
- **GPU/CPU Compatible**: Automatic device detection for optimal performance

## 📋 Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- NumPy
- tqdm

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Tanjim-Islam/MultiLang-NER-BERT.git
cd MultiLang-NER-BERT
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset

### Dataset Link:
```
https://drive.google.com/drive/folders/1XUIjlkjdW42e9md6zODt_FEBsBlL9HME
```

### Dataset Structure
The project expects data in CoNLL format with the following directory structure:
```
train_dev/
├── {LANGUAGE_CODE}-train.conll
└── {LANGUAGE_CODE}-dev.conll
```

### Supported Formats
- **CoNLL Format**: Each line contains a token and its corresponding NER label, separated by whitespace
- **Empty Lines**: Separate sentences/documents
- **Example**:
```
John B-PER
works O
at O
Google B-ORG
. O

Mary B-PER
lives O
in O
New B-LOC
York I-LOC
. O
```

## 🔧 Configuration

The main configuration parameters can be modified in the notebook:

```python
MAX_LEN = 128          # Maximum sequence length
BATCH_SIZE = 32        # Training batch size
LEARNING_RATE = 2e-5   # Learning rate for AdamW optimizer
NUM_EPOCHS = 5         # Number of training epochs
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LANGUAGE_CODE = 'fr'   # Language code for dataset files
```

### Supported Languages
The model supports any language included in the multilingual BERT vocabulary. Common language codes include:
- `en` - English
- `fr` - French
- `de` - German
- `es` - Spanish
- `it` - Italian
- `pt` - Portuguese
- `nl` - Dutch
- `zh` - Chinese
- And many more...

## 🚀 Usage

1. **Prepare your data**: Ensure your CoNLL formatted files are in the `train_dev/` directory
2. **Set language code**: Change the `LANGUAGE_CODE` variable to your target language
3. **Configure parameters**: Adjust the configuration variables as needed
4. **Run the notebook**: Execute all cells in `main.ipynb`

### Step-by-Step Process

The notebook follows this workflow:

1. **Data Loading**: Reads CoNLL formatted files and parses tokens/labels
2. **Tokenization**: Uses BERT tokenizer to convert text to model input
3. **Label Encoding**: Maps NER labels to numerical IDs
4. **Model Initialization**: Loads pre-trained multilingual BERT for token classification
5. **Training**: Fine-tunes the model on your dataset
6. **Evaluation**: Computes performance metrics on validation set

## 🏗️ Model Architecture

- **Base Model**: `bert-base-multilingual-cased`
- **Task**: Token Classification (Named Entity Recognition)
- **Tokenizer**: `BertTokenizerFast` with multilingual support
- **Optimizer**: AdamW with learning rate 2e-5
- **Loss Function**: Cross-entropy loss (built into `BertForTokenClassification`)

## 📈 Performance

Example results on French NER dataset:

| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | 76.92% | 77.17% |
| Precision | 59.25% | 59.55% |
| Recall | 76.92% | 77.17% |
| F1-Score | 66.94% | 67.23% |

*Results may vary depending on the dataset and language used.*

## 🔍 Functions Overview

### Core Functions

- `load_data(file_path)`: Loads CoNLL formatted data and returns tokens and labels
- `encode_tags(tags, encodings)`: Aligns NER labels with BERT subword tokens
- `create_dataset(token_lists, tag_lists)`: Creates PyTorch dataset from tokens and labels

### Key Features of Implementation

- **Subword Alignment**: Properly handles BERT's subword tokenization
- **Padding and Truncation**: Ensures consistent sequence lengths
- **Label Masking**: Uses -100 for non-first subword tokens (following BERT conventions)
- **Progress Tracking**: Uses tqdm for training progress visualization

## 🎯 Customization

### Adding New Languages
1. Obtain CoNLL formatted data for your target language
2. Place files in `train_dev/` directory with naming convention: `{lang}-train.conll`, `{lang}-dev.conll`
3. Update `LANGUAGE_CODE` variable
4. Run the training pipeline

### Hyperparameter Tuning
- **MAX_LEN**: Increase for longer sequences, decrease for memory efficiency
- **BATCH_SIZE**: Adjust based on GPU memory availability
- **LEARNING_RATE**: Typical range: 1e-5 to 5e-5 for BERT fine-tuning
- **NUM_EPOCHS**: Increase for better convergence, watch for overfitting

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `BATCH_SIZE` or `MAX_LEN`
2. **File Not Found**: Ensure CoNLL files are in correct directory with proper naming
3. **Poor Performance**: Check data quality, increase training epochs, or adjust learning rate
4. **Label Mismatch**: Verify CoNLL format correctness and label consistency

### Memory Optimization Tips

- Use gradient accumulation for larger effective batch sizes
- Enable mixed precision training with `torch.cuda.amp`
- Consider using smaller sequence lengths for memory-constrained environments

## 📄 License

This project is open-source. Please check the repository for license details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📚 References

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [CoNLL-2003 Shared Task](https://www.clips.uantwerpen.be/conll2003/ner/)

---

**Note**: Change the "LANGUAGE_CODE" to the language you want to train and adjust the "Configuration" parameters as needed for your specific use case.
