# Attention Is All You Need - Transformer Model

This repository contains a from-scratch implementation of the Transformer architecture, originally introduced in the paper [“Attention Is All You Need”](https://arxiv.org/abs/1706.03762), using PyTorch.

The model includes:
- Multi-head self-attention
- Positional encoding
- Encoder-decoder architecture
- Masked decoding
- Layer normalization, residual connections, dropout

## Files
- `main.py` — Full implementation of the Transformer including `SelfAttention`, `TransformerBlock`, `Encoder`, `DecoderBlock`, `Decoder`, and the full `Transformer` class.

## How to Run

```bash
# Install requirements
pip install torch

# Run the script
python main.py


## Credits

This Transformer model implementation is based on the excellent tutorial by [Aladdin Persson](https://www.youtube.com/c/AladdinPersson):

- 📺 YouTube: [The Transformer Neural Network Explained](https://youtu.be/U0s0f995w14)
- 🧠 GitHub: [Machine Learning Collection](https://github.com/aladdinpersson/Machine-Learning-Collection)
- 📝 License: MIT

I thank Aladdin for his clear and educational content!
