import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.linear = nn.Linear(input_dim, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        return self.linear(x)  # (batch_size, seq_len, d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create a long enough positional encoding once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as a buffer so it's not trained
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # Add position encoding
        x = x + self.pe[:, :seq_len, :]
        return x

class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        return self.layer_norm(x)

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()  # or nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, nheads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nheads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # query, key, value shapes: (batch_size, seq_len, d_model)
        out, _ = self.mha(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.dropout(out)

class ResidualConnection(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = LayerNormalization(d_model)

    def forward(self, x, sublayer_out):
        # sublayer_out is the output of either attention or feed-forward
        return self.norm(x + sublayer_out)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionBlock(d_model, nheads, dropout)
        self.res1 = ResidualConnection(d_model)

        self.ff = FeedForwardBlock(d_model, dim_feedforward, dropout)
        self.res2 = ResidualConnection(d_model)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # 1) Self-attention
        attn_out = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        x = self.res1(x, attn_out)

        # 2) Feed-forward
        ff_out = self.ff(x)
        x = self.res2(x, ff_out)

        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, nheads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, nheads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask, src_key_padding_mask)
        return x

class Projection3D(nn.Module):
    """
    A custom layer that maps (B, d_model) -> (B, future_len, output_dim)
    by learning a weight of shape (d_model, future_len, output_dim).
    """
    def __init__(self, d_model, future_len, output_dim):
        super().__init__()
        # weights in a 3D parameter: (d_model, future_len, output_dim)
        self.weight = nn.Parameter(torch.randn(d_model, future_len, output_dim))
        # 2D bias: (future_len, output_dim)
        self.bias   = nn.Parameter(torch.zeros(future_len, output_dim))

    def forward(self, x):
        """
        x is (B, d_model)
        output is (B, future_len, output_dim)
        out[b, i, j] = sum_k( x[b, k] * weight[k, i, j] ) + bias[i, j]
        """
        # torch.einsum allows a clean expression of the above multiplication:
        # 'bd,dfg->bfg':
        #  b = batch
        #  d = dimension in the input (d_model)
        #  f = future_len
        #  g = output_dim
        out = torch.einsum('bd,dfg->bfg', x, self.weight) + self.bias
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        d_model=512,
        nheads=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        future_len=15,
        output_dim=2
    ):
        super().__init__()
        self.future_len = future_len
        self.output_dim = output_dim

        # Embedding for input
        self.encoder_embedding = InputEmbeddings(input_dim, d_model)
        self.encoder_pe        = PositionalEncoding(d_model)

        # Transformer encoder
        self.encoder = Encoder(num_encoder_layers, d_model, nheads, dim_feedforward, dropout)

        # Custom 3D projection, output: (B, future_len, output_dim) E.g: (4, 15, 2) - Speedoinfo and SteerwheelAngle
        self.predictor = Projection3D(d_model, future_len, output_dim)

    def forward(self, src):
        """
        src shape: (B, src_seq_len=25, input_dim=1024)
        Return shape: (B, future_len=15, 2)
        """
        # 1) Embed + Pos Encode
        x = self.encoder_embedding(src)
        x = self.encoder_pe(x)

        # 2) Pass through the encoder
        encoded = self.encoder(x)  # (B, 25, d_model)

        # 2 Options: the last timestep's hidden state or do mean-pooling
        
	# Option A: last hidden state
        last_h = encoded[:, -1, :]  # (B, d_model)

	# Option B: mean-pooling of encoder outputs
	# pooled = encoded.mean(dim=1)  # (B, d_model


        # 3) Map to future predictions
        out = self.predictor(last_h)  # out: (B, 15*2)
	#out = self.predictor(pooled) # for Option B
	
        return out

# --------------------------------------------------------------------------------------
# 			DUMMY DATA TO VERIFY CLASS OUTPUT
# --------------------------------------------------------------------------------------
'''
import torch

model = Transformer(
    input_dim=1024,
    d_model=512,
    nheads=8,
    num_encoder_layers=8,
    dim_feedforward=2048,
    dropout=0.2,
    future_len=15,          # predict 15 timestep steps
    output_dim=2            # speed + steer
)

batch_size = 2
src_seq_len = 25
input_dim = 1024

dummy_src = torch.randn(batch_size, src_seq_len, input_dim)
print("Dummy src shape:", dummy_src.shape)

dummy_out = model(dummy_src)

print("Model output shape:", dummy_out.shape)

'''

