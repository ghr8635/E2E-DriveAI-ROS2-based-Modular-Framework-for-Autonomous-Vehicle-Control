import torch
import torch.nn as nn
import math


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, seq_length, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()

        # Embedding Layer
        self.embedding = nn.Linear(input_dim, d_model)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, seq_length)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Additional Feed-Forward Layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, output_dim)

        # Activation and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Embedding
        x = self.embedding(x)

        # Apply Positional Encoding
        x = self.positional_encoding(x)

        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)

        # Take the output of the last timestep
        x = x[:, -1, :]

        # Additional Feed-Forward Layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Define Positional Encoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix for positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine and cosine functions to alternate dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it’s part of the model but not learnable
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)