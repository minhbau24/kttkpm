import torch
import torch.nn as nn


class SequentialRecModel(nn.Module):
    def __init__(
        self,
        vocab_sizes,
        embed_dims=None,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=False,
        rnn_type="lstm",
        proj_dim=64,
    ):
        super().__init__()
        embed_dims = embed_dims or {
            "product": 32,
            "action": 8,
            "category": 16,
            "price": 8,
            "seller": 12,
        }

        self.product_embed = nn.Embedding(vocab_sizes["product"], embed_dims["product"], padding_idx=0)
        self.action_embed = nn.Embedding(vocab_sizes["action"], embed_dims["action"], padding_idx=0)
        self.category_embed = nn.Embedding(vocab_sizes["category"], embed_dims["category"], padding_idx=0)
        self.price_embed = nn.Embedding(vocab_sizes["price"], embed_dims["price"], padding_idx=0)
        self.seller_embed = nn.Embedding(vocab_sizes["seller"], embed_dims["seller"], padding_idx=0)

        input_dim = sum(embed_dims.values()) + 1
        self.input_proj = nn.Linear(input_dim, proj_dim)

        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        output_dim = hidden_size * (2 if bidirectional else 1)
        self.product_head = nn.Linear(output_dim, vocab_sizes["product"])
        self.action_head = nn.Linear(output_dim, vocab_sizes["action"])
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()

    def forward(self, batch):
        prod = self.product_embed(batch["product_seq"])
        action = self.action_embed(batch["action_seq"])
        category = self.category_embed(batch["category_seq"])
        price = self.price_embed(batch["price_seq"])
        seller = self.seller_embed(batch["seller_seq"])

        time_delta = batch["time_delta_seq"].unsqueeze(-1)
        x = torch.cat([prod, action, category, price, seller, time_delta], dim=-1)
        x = self.activation(self.input_proj(x))
        x = self.dropout(x)

        if self.rnn_type == "lstm":
            rnn_out, (h_n, _) = self.rnn(x)
        else:
            rnn_out, h_n = self.rnn(x)

        if self.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            last_hidden = h_n[-1]

        product_logits = self.product_head(last_hidden)
        action_logits = self.action_head(last_hidden)
        return {
            "product_logits": product_logits,
            "action_logits": action_logits,
        }


def build_lstm_model(vocab_sizes, **kwargs):
    return SequentialRecModel(vocab_sizes=vocab_sizes, rnn_type="lstm", bidirectional=False, **kwargs)


def build_gru_model(vocab_sizes, **kwargs):
    return SequentialRecModel(vocab_sizes=vocab_sizes, rnn_type="gru", bidirectional=False, **kwargs)


def build_bilstm_model(vocab_sizes, **kwargs):
    return SequentialRecModel(vocab_sizes=vocab_sizes, rnn_type="lstm", bidirectional=True, **kwargs)
