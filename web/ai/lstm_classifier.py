import torch
import torch.nn as nn

class PurchaseSeqClassifier(nn.Module):
    def __init__(self, vocab_size=8, embed_dim=16, hidden_size=32, num_layers=1, dropout=0.3, rnn_type="lstm"):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")
            
        self.dropout = nn.Dropout(dropout)
        
        # Classifier head
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        embeds = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        if self.rnn_type == "lstm":
            rnn_out, (h_n, c_n) = self.rnn(embeds)
        else:
            rnn_out, h_n = self.rnn(embeds)
            
        # Extract the last hidden state of the final RNN layer
        # h_n shape: (num_layers, batch_size, hidden_size)
        last_hidden = h_n[-1]  # (batch_size, hidden_size)
        
        # Fully connected layers
        out = self.fc1(self.dropout(last_hidden))
        out = self.relu(out)
        logits = self.fc2(self.dropout(out))  # (batch_size, 1)
        
        return logits.squeeze(1)  # (batch_size,)
