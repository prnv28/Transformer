# Neural Machine Translation using Transformer | English to Hindi

This repository contains an **implementation from scratch** of the Transformer architecture for **Neural Machine Translation (NMT)**. The model is trained to translate sentences from **English to Hindi**, based on the **IIT Bombay English-Hindi Parallel Corpus**.

> 📜 Paper Reference: "Attention is All You Need" by Vaswani et al. (2017)  
> 🔗 GitHub: [https://github.com/prnv28/Transformer](https://github.com/prnv28/Transformer)

---

## 📌 Project Highlights

- Full implementation of the **Transformer model** using PyTorch
- Training on the **IITB English-Hindi parallel dataset**
- Complete preprocessing pipeline including tokenization, vocabulary creation, and padding
- Support for both **training and inference**
- Option to evaluate BLEU scores for model performance

---

## 🗂️ Project Structure

```
└── Transformer/
    └── src
        ├── Inference.ipynb
        ├── config.py
        ├── dataset.py
        ├── model.py
        ├── requirements.txt
        ├── runs
        ├── tokenizer_en.json
        ├── tokenizer_hi.json
        ├── tokenizer_it.json
        ├── train.py
        └── translate.py
```

---
## 🛠️ Implementation Details

### Model Architecture
| Component              | Specification                          |
|------------------------|----------------------------------------|
| Embedding Dimension    | 512                                    |
| Feed Forward Dimension | 2048                                   |
| Attention Heads        | 8                                      |
| Encoder/Decoder Layers | 6                                      |
| Dropout                | 0.1                                    |

### Training
- Optimizer: Adam (β₁=0.9, β₂=0.98, ε=10⁻⁹)
- Learning Rate: Custom schedule (warmup steps = 4000)
- Batch Size: 32
- Training epochs: 15
---

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/prnv28/Transformer.git
cd Transformer
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```
---

## 🚀 Training

To train the model from scratch:

```bash
python train.py
```

You can adjust hyperparameters like batch size, learning rate, and number of epochs inside `config.py`.


---

## 📚 References

- Vaswani et al., "Attention is All You Need", NeurIPS 2017 [[paper](https://arxiv.org/abs/1706.03762)]
- IIT Bombay English-Hindi Parallel Corpus [[link](https://www.cfilt.iitb.ac.in/iitb_parallel/)]

---
