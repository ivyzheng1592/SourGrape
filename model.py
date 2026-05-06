import torch
from torch import nn

from hyper_params import HyperParams


class PhonemeRegressor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = HyperParams().embed_size,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.head = nn.Linear(embed_size, 1)

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.embedding(phoneme_ids)
        out = self.head(emb).squeeze(-1)
        return out


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        embed_size: int = HyperParams().embed_size,
        hidden_size: int = HyperParams().hidden_size,
        num_layers: int = HyperParams().num_layers,
        dropout: float = HyperParams().dropout,
        embedding_weights: torch.Tensor | None = None,
        freeze_embedding: bool = False,
    ) -> None:
        super().__init__()
        # Character embeddings; padding index 0 matches PAD_TOKEN.
        if embedding_weights is None:
            self.embedding = nn.Embedding(
                input_size,
                embed_size,
                padding_idx=HyperParams().padding_id,
            )
        else:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_weights,
                freeze=freeze_embedding,
                padding_idx=HyperParams().padding_id,
            )
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

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Embed characters into vectors.
        emb = self.embedding(x)
        # Run the LSTM across the fixed-length word.
        out, _ = self.lstm(emb)
        # Use the final timestep output as the word representation.
        last_hidden = out[:, -1, :]
        # Map to trajectory prediction.
        return self.head(last_hidden)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.score_proj = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Score each encoder state against the current decoder state.
        score_input = torch.tanh(
            self.query_proj(decoder_hidden).unsqueeze(1) + self.key_proj(encoder_outputs)
        )
        scores = self.score_proj(score_input).squeeze(-1)
        scores = scores.masked_fill(~encoder_mask, -1e9)
        attn_weights = torch.softmax(scores, dim=1)
        # Form the context vector as the weighted sum of encoder states.
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class Seq2SeqRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_len: int,
        embed_size: int = HyperParams().embed_size,
        hidden_size: int = HyperParams().hidden_size,
        num_layers: int = HyperParams().num_layers,
        dropout: float = HyperParams().dropout,
        embedding_weights: torch.Tensor | None = None,
        freeze_embedding: bool = False,
        padding_id: int = HyperParams().padding_id,
        padding_value: float = HyperParams().padding_value,
        teacher_forcing_ratio: float = HyperParams().teacher_forcing_ratio,
    ) -> None:
        super().__init__()
        self.output_len = output_len
        self.padding_id = padding_id
        self.padding_value = padding_value
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Embed the input characters.
        if embedding_weights is None:
            self.embedding = nn.Embedding(
                input_size,
                embed_size,
                padding_idx=HyperParams().padding_id,
            )
        else:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_weights,
                freeze=freeze_embedding,
                padding_idx=HyperParams().padding_id,
            )
        # Encode the input sequence.
        self.encoder = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Score the encoder outputs at each decoder step.
        self.attention = BahdanauAttention(hidden_size)
        # Update the decoder state one step at a time.
        self.decoder = nn.LSTMCell(
            input_size=hidden_size + 1,
            hidden_size=hidden_size,
        )
        # Predict the next trajectory value.
        self.out_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Encode the input sequence.
        emb = self.embedding(x)
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(emb)
        
        # Mark the non-padding encoder positions.
        encoder_mask = x != self.padding_id
        # Initialize the decoder state from the last encoder states.
        decoder_hidden = encoder_hidden[-1]
        decoder_cell = encoder_cell[-1]
        
        # Use zeros as the first decoder input.
        decoder_input = torch.zeros(x.size(0), 1, dtype=emb.dtype, device=emb.device)
        # Replace padded target values before teacher forcing uses them.
        if targets is not None:
            targets = targets.masked_fill(targets == self.padding_value, 0.0)
        
        batch_size = x.size(0)
        outputs = []
        for step in range(self.output_len):
            # Read a context vector from the encoder outputs.
            context, _ = self.attention(decoder_hidden, encoder_outputs, encoder_mask)
            lstm_input = torch.cat([decoder_input, context], dim=1)
            
            # Update the decoder state.
            decoder_hidden, decoder_cell = self.decoder(
                lstm_input,
                (decoder_hidden, decoder_cell),
            )
            decoder_output = self.out_proj(
                torch.cat([decoder_hidden, context], dim=1)
            )
            outputs.append(decoder_output)
            
            # Select the next decoder input.
            if targets is not None:
                # Choose whether to use the target value or the predicted value.
                teacher_mask = (
                    torch.rand(batch_size, 1, device=x.device) < self.teacher_forcing_ratio
                )
                teacher_value = targets[:, step].unsqueeze(1)
                decoder_input = torch.where(teacher_mask, teacher_value, decoder_output)
            else:
                decoder_input = decoder_output

        return torch.cat(outputs, dim=1)
