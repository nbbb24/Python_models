# Transformer Implementation

A PyTorch implementation of the Transformer architecture as described in "Attention is All You Need".

## Project Structure
```
Transformer/
├── layers/           # Core building blocks of the Transformer
│   ├── multi_head_attention.py
│   ├── positional_encoding.py
│   └── README.md
├── requirements.txt  # Project dependencies
└── README.md        # This file
```

## Core Components

### Multi-Head Attention
<img src="Images/MHA.png" width="45%"/> <img src="Images/scale_dot_product.png" width="45%"/>

### Positional Encoding
The positional encoding module adds information about the position of each token in the sequence. It uses sine and cosine functions of different frequencies to create unique patterns for each position:

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos` is the position in the sequence
- `i` is the dimension index
- `d_model` is the dimension of the model embeddings

## Documentation
Each directory contains its own README.md file with detailed information about its contents and purpose. 