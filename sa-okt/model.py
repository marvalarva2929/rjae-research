
import torch.nn as nn
import torch
import torch.nn.functional as F


class SAOKT(nn.Module):
    def __init__(self, qdim, max_seq_length, num_heads,
                 num_apis, dropout, binary_out=False):
        
        super().__init__()

        kdim = 3 * qdim
        self.max_seq_length = max_seq_length
        self.positional_embedding = nn.Embedding(
                num_embeddings=max_seq_length+1, embedding_dim=kdim)

        self.question_embedding = None  # GPT Encoder (text -> d)
        self.response_embedding = nn.Embedding(num_embeddings=num_apis,
                                               embedding_dim=qdim)
        self.answer_embedding = nn.Embedding(num_embeddings=num_apis,
                                             embedding_dim=qdim)

        self.matn = nn.MultiheadAttention(
                embed_dim=qdim, num_heads=num_heads, dropout=dropout,
                kdim=kdim, vdim=kdim)

        self.ff1 = nn.Linear(in_features=qdim, out_features=qdim)

        self.prediction = nn.Linear(in_features=qdim,
                                    out_features=max_seq_length)

        self.layernorm = nn.LayerNorm(qdim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, question_sequence,
                response_sequence, answer_sequence, is_embedded=True):

        if not is_embedded:
            question_sequence = self.question_embedding(question_sequence)
            response_sequence = self.response_embedding(response_sequence)
        answer_sequence = self.response_embedding(answer_sequence)

        query = question_sequence
        
        print(response_sequence.shape, question_sequence.shape, answer_sequence.shape)

        x = torch.cat((question_sequence, response_sequence,
                      answer_sequence), dim=-1)

        x += self.positional_embedding(torch.arange(len(x)))

        x = self.dropout(x)
        future_mask = None
        res, _ = self.matn(query=query, key=x, value=x)
        out = self.dropout(F.relu(self.ff1(res)))

        out += self.layernorm(res)
        out = self.layernorm(out)

        logits = self.prediction(out)
        return logits
