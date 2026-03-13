
import torch
import torch.nn as nn
import numpy as np

def BoaGRU(d_model=256, num_layers=4, vocab_size=256, device="cuda"):
    """ Construct a BoaBytePredictor with GRU """


    class BoaBytePredictorGRU(nn.Module):
        """ GRU model adapted to predict the next byte in a sequence. """

        def __init__(self, d_model=256, num_layers=4, vocab_size=256):
            super().__init__()
            # Embedding for vocab_size possible bytes
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=num_layers, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                # Output logits for each of the vocab_size possible next bytes
                nn.Linear(d_model, vocab_size)
            )
            self.d_model= d_model
            self.num_layers = num_layers

        def forward(self, x):
            h = self.embedding(x)  # [B, L, D]
            output, _ = self.gru(h)
            return self.head(output)

        @torch.inference_mode()
        def init_stream(self, max_len: int, batch_size: int = 1, device=None, dtype=None):
              h_0 = torch.zeros(self.num_layers, batch_size, self.d_model, device=device, dtype=dtype)
              return [h_0]

        @torch.inference_mode()
        def step(self, byte_t: torch.LongTensor, caches) -> torch.Tensor:
                # byte_t: [B] -> logits: [B, 256]
              x = self.embedding(byte_t).unsqueeze(1)  # [B, 1, D]
              prev_states = caches[0]
              gru_out, new_states = self.gru(x, prev_states)

              caches[0] = new_states

              logits = self.head(gru_out)
              return logits.squeeze(1)

    model = BoaBytePredictorGRU(d_model, num_layers, vocab_size)
    return model.to(device)
