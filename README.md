# CLAP: Compact Linearization with an Adaptable Parser

Welcome to the official repository for [*CLAP*](https://aclanthology.org/2024.lrec-main.495/), an innovative architecture for AMR (Abstract Meaning Representation) parsing, presented at LREC-COLING 2024.

## Features

1. **AMR Parsing and Generation**: CLAP introduces a flexible and efficient AMR parsing architecture. It supports seamless transitions between different language models and facilitates multilingual adaptability.

2. **Crosslingual AMR Alignment**: Integration of the [*Crosslingual AMR Aligner*](https://aclanthology.org/2023.findings-acl.109/) enables extraction of span-to-node alignments from sentences to graphs, leveraging the model's cross-attention capabilities.

3. **Perplexity Extraction**: Incorporating the [*AMRs Assemble*](https://aclanthology.org/2023.findings-acl.109/), CLAP can compute perplexity scores and supports training in assembly tasks.

## Citing This Work

If you use CLAP in your research, please cite our paper:

```bibtex
@inproceedings{martinez-lorenzo-navigli-2024-efficient-amr,
    title = "Efficient {AMR} Parsing with {CLAP}: Compact Linearization with an Adaptable Parser",
    author = "Martinez Lorenzo, Abelardo Carlos and Navigli, Roberto",
    editor = "Calzolari, Nicoletta and Kan, Min-Yen and Hoste, Veronique and Lenci, Alessandro and Sakti, Sakriani and Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.495",
    pages = "5578--5584",
}
```

## Repository Structure

- `conf/`: Configuration files for data paths, model specifications, and training parameters.
- `data/`: Datasets for benchmarking AMR evaluation.
- `experiments/`: Stores checkpoints post-training.
- `models/`: Trained Hugging Face models.
- `src/`: Source code for the project.
  - `constant.py`: Manages tokens added to the model; customizable for new tokens.
  - `linearization.py`: Implements graph linearization in Depth-First Search and compact formats.
  - `pl_data_modules.py`: Data module classes for training.
  - `pl_modules.py`: Contains new modular components for the architecture.
  - `predict.py`: Script for making predictions using trained models.
  - `predict_alignment.py`: Script for extracting alignments.
  - `predict_perplexity.py`: Script for computing perplexity.
  - `train.py`: Entry point for training models.
  - `utils.py`: Utility functions for various operations.

## Installation

```bash
# Create a Python 3.9 environment
conda create -n clap-env python=3.9
conda activate clap-env

# Install dependencies
pip install -r requirements.txt
```



## Training 

Configure paths and hyperparameters in conf/ directory files:

- conf/data.yaml: Specify dataset paths for training and evaluation.
- conf/model.yaml: Define the model architecture, e.g., google/flan-t5-small.
- conf/train.yaml: Adjust training-specific hyperparameters.

```bash
python src/train.py
```

## Prediction

Set up the necessary paths in conf/data.yaml and conf/model.yaml. Then run:

```bash
python src/predict.py
```

# Alignment Extraction

Configure as per the prediction step and execute:

```bash
python src/predict_alignments.py
```

# Perplexity Calculation

Configure as per the prediction step and execute:

```bash
python src/predict_perplexity.py
```

## License
This project is released under the CC-BY-NC-SA 4.0 license (see `LICENSE`). If you use AMRs-Assemble!, please reference the paper and put a link to this repo.

## Contributing

We welcome contributions to the Cross-lingual AMR Aligner project. If you have any ideas, bug fixes, or improvements, feel free to open an issue or submit a pull request.

## Contact

For any questions or inquiries, please contact Abelardo Carlos Martínez Lorenzo at martineslorenzo@diag.uniroma.it
