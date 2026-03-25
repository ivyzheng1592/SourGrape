import torch
from torch import nn


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        embed_size: int = 16,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        # Character embeddings; padding index 0 matches PAD_TOKEN.
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        # PyTorch applies dropout between LSTM layers only when num_layers > 1.
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        # Regression head predicts the full trajectory vector.
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed characters into vectors.
        emb = self.embedding(x)
        # Run the LSTM across the fixed-length word.
        out, _ = self.lstm(emb)
        # Use the final timestep output as the word representation.
        last_hidden = out[:, -1, :]
        # Map to trajectory prediction.
        return self.head(last_hidden)


class Seq2SeqRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_len: int,
        embed_size: int = 16,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        # Encoder maps characters to a hidden state.
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        enc_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=enc_dropout,
        )
        # Decoder consumes a zero input sequence and unfolds the trajectory.
        self.output_len = output_len
        dec_dropout = dropout if num_layers > 1 else 0.0
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dec_dropout,
        )
        # Map decoder hidden states to scalar trajectory values.
        self.out_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode the input word to initialize the decoder.
        emb = self.embedding(x)
        _, (h_n, c_n) = self.encoder(emb)
        # Use a zero input sequence for the decoder steps.
        batch_size = x.size(0)
        dec_in = torch.zeros(
            batch_size,
            self.output_len,
            1,
            dtype=emb.dtype,
            device=emb.device,
        )
        dec_out, _ = self.decoder(dec_in, (h_n, c_n))
        # Project to trajectory values and squeeze the feature dimension.
        y = self.out_proj(dec_out).squeeze(-1)
        return y
