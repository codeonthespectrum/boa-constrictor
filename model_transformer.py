import torch
import torch.nn as nn
import numpy as np
import math

def BoaTransformer(d_model=256, num_layers=1, vocab_size=256, device="cuda"):
    """Construct a BoaBytePredictor with Transformer decoder-only backbone."""

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=65536):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, D]

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class BoaBytePredictorTransformer(nn.Module):
        def __init__(self, d_model=256, num_layers=1, vocab_size=256, max_context=1024):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_enc = PositionalEncoding(d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, dim_feedforward=4*d_model,
                dropout=0.0, batch_first=True, activation='gelu'
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, vocab_size)
            )
            self.d_model = d_model
            self.num_layers = num_layers
            self.max_context = max_context

        def _causal_mask(self, seq_len, device):
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            return mask.masked_fill(mask == 1, float('-inf'))

        def forward(self, x):
            h = self.embedding(x)
            h = self.pos_enc(h)
            mask = self._causal_mask(x.size(1), x.device)
            h = self.transformer(h, mask=mask)
            return self.head(h)

        @torch.inference_mode()
        def init_stream(self, max_len: int, batch_size: int = 1, device=None, dtype=None):
            # Buffer to store previous tokens
            return [torch.zeros(batch_size, 0, dtype=torch.long, device=device)]

        @torch.inference_mode()
        def step(self, byte_t: torch.LongTensor, caches) -> torch.Tensor:
            buffer = caches[0]
            # Append new token to buffer
            buffer = torch.cat([buffer, byte_t.unsqueeze(1)], dim=1)
            # Limit context window
            if buffer.size(1) > self.max_context:
                buffer = buffer[:, -self.max_context:]
            caches[0] = buffer
            # Run full forward on buffer, take last position
            logits = self.forward(buffer)
            return logits[:, -1, :]

    model = BoaBytePredictorTransformer(d_model=d_model, num_layers=num_layers, vocab_size=vocab_size)
    return model.to(device)
