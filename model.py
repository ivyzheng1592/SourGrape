import torch
from torch import nn

import hyper_params as hp


class PhonemeRegressor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = hp.embed_size,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.out_proj = nn.Linear(embed_size, 1)

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.embedding(phoneme_ids)
        out = self.out_proj(emb).squeeze(-1)
        return out


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        embed_size: int = hp.embed_size,
        hidden_size: int = hp.hidden_size,
        num_layers: int = hp.num_layers,
        dropout: float = hp.dropout,
        bidirectional: bool = hp.bidirectional,
        embedding_weights: torch.Tensor | None = None,
        freeze_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        # Character embeddings; padding index 0 matches PAD_TOKEN.
        if embedding_weights is None:
            self.embedding = nn.Embedding(
                input_size,
                embed_size,
                padding_idx=hp.padding_id,
            )
        else:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_weights,
                freeze=freeze_embedding,
                padding_idx=hp.padding_id,
            )
        # Encode the full character sequence into the final hidden state(s).
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Regression head predicts the full trajectory vector.
        self.out_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * (2 if bidirectional else 1), output_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Embed characters into vectors.
        emb = self.embedding(x)
        # Run the LSTM across the fixed-length word.
        _, (hidden, _) = self.lstm(emb)
        # Use the last encoder states as the word representation.
        hidden = hidden.view(self.num_layers, self.num_directions, x.size(0), self.hidden_size)
        last_hidden = hidden[-1].transpose(0, 1).reshape(x.size(0), -1)
        # Map to trajectory prediction.
        return self.out_proj(last_hidden)


class BahdanauAttention(nn.Module):
    def __init__(self, decoder_hidden_size: int, encoder_output_size: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(decoder_hidden_size, decoder_hidden_size, bias=False)
        self.key_proj = nn.Linear(encoder_output_size, decoder_hidden_size, bias=False)
        self.score_proj = nn.Linear(decoder_hidden_size, 1, bias=False)

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
        embed_size: int = hp.embed_size,
        hidden_size: int = hp.hidden_size,
        num_layers: int = hp.num_layers,
        dropout: float = hp.dropout,
        bidirectional: bool = hp.bidirectional,
        embedding_weights: torch.Tensor | None = None,
        freeze_embedding: bool = False,
        padding_id: int = hp.padding_id,
        padding_value: float = hp.padding_value,
        teacher_forcing_ratio: float = hp.teacher_forcing_ratio,
    ) -> None:
        super().__init__()
        self.output_len = output_len
        self.padding_id = padding_id
        self.padding_value = padding_value
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.encoder_output_size = hidden_size * self.num_directions

        # Embed the input characters.
        if embedding_weights is None:
            self.embedding = nn.Embedding(
                input_size,
                embed_size,
                padding_idx=hp.padding_id,
            )
        else:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_weights,
                freeze=freeze_embedding,
                padding_idx=hp.padding_id,
            )
        # Encode the input sequence.
        self.encoder = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Score the encoder outputs at each decoder step.
        self.attention = BahdanauAttention(hidden_size, self.encoder_output_size)
        self.encoder_hidden_proj = nn.Linear(self.encoder_output_size, hidden_size)
        self.encoder_cell_proj = nn.Linear(self.encoder_output_size, hidden_size)
        # Update the decoder state one step at a time.
        self.decoder = nn.LSTMCell(
            input_size=self.encoder_output_size + 1,
            hidden_size=hidden_size,
        )
        # Predict the next trajectory value.
        self.out_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size + self.encoder_output_size, 1),
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
        encoder_hidden = encoder_hidden.view(
            self.num_layers,
            self.num_directions,
            x.size(0),
            self.hidden_size,
        )
        encoder_cell = encoder_cell.view(
            self.num_layers,
            self.num_directions,
            x.size(0),
            self.hidden_size,
        )
        last_hidden = encoder_hidden[-1].transpose(0, 1).reshape(x.size(0), -1)
        last_cell = encoder_cell[-1].transpose(0, 1).reshape(x.size(0), -1)
        decoder_hidden = self.encoder_hidden_proj(last_hidden)
        decoder_cell = self.encoder_cell_proj(last_cell)
        
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
