import math
import torch
import torch.nn as nn


# to understand more about the transformer, please refer to:
# https://jalammar.github.io/illustrated-transformer/
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        density = torch.exp(-torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * density)
        pos_embedding[:, 1::2] = torch.cos(pos * density)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, x):
        x = x + self.pos_embedding[: x.size(0), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.sqrt_embedding_dim = math.sqrt(embedding_dim)

    def forward(self, x):
        x = self.embedding(x.long())
        # Multiply sqrt of embedding dim to stabilize the variance
        # and avoid the gradient vanishing or exploding problem
        return x * self.sqrt_embedding_dim


# https://pytorch.org/tutorials/beginner/translation_transformer.html
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        source_vocab_size,
        target_vocab_size,
        nhead,
        num_layers,
        feedforward_dim,
        dropout,
    ):
        super().__init__()
        self.tansformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(embedding_dim, target_vocab_size)
        self.source_token_embedding = TokenEmbedding(source_vocab_size, embedding_dim)
        self.target_token_embedding = TokenEmbedding(target_vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)

    def _embedding_with_position(self, x, embedding: TokenEmbedding):
        # shape: (batch_size, seq_len)
        x = embedding(x)
        # shape: (batch_size, seq_len, embedding_dim)
        x = self.positional_encoding(x)
        # shape: (batch_size, seq_len, embedding_dim)
        return x

    def forward(
        self,
        source,
        target,
        source_mask,
        target_mask,
        source_padding_mask,
        target_padding_mask,
        memory_key_padding_mask,
    ):
        # shape: (batch_size, source_seq_len, embedding_dim)
        source = self._embedding_with_position(source, self.source_token_embedding)
        # shape: (batch_size, target_seq_len, embedding_dim)
        target = self._embedding_with_position(target, self.target_token_embedding)

        # MultiHeadAttention in Transformer could handle the different sequence length
        # in different batches of the source and target
        # In typically implementation of attention, we need to set the max length of the sequence
        x = self.tansformer(
            source,
            target,
            src_mask=source_mask,
            tgt_mask=target_mask,
            memory_mask=None,
            src_key_padding_mask=source_padding_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        # shape: (batch_size, target_seq_len, embedding_dim)
        x = self.generator(x)
        # shape: (batch_size, target_seq_len, target_vocab_size)
        # output the possibility of each token in the target vocab
        return x

    # with encode the source into key/value memory for future decode
    def encode(self, source):
        x = self._embedding_with_position(source, self.source_token_embedding)
        return self.tansformer.encoder(x)

    # decode the target as query with the key/value memory from the encoder
    def decode(self, target, memory, target_mask):
        target = self._embedding_with_position(target, self.target_token_embedding)
        decoded = self.tansformer.decoder(target, memory, target_mask)
        # the decoded result which shape is (batch_size, target.size(0), embedding_dim)
        # generate the possibility of each token in the target vocab
        # prob shape is (1, target_vocab_size)
        prob = self.generator(decoded[:, -1])
        # get the next token
        _, next_token = torch.max(prob, dim=1)
        # from torch.Size([1]) to torch.Size([1, 1])
        return next_token.unsqueeze(0)
