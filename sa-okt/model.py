
import torch.nn as nn
import torch
import torch.nn.functional as F


class SAOKT(nn.Module):
    def __init__(self, latent_dim, max_seq_length, num_heads,
                 num_apis, dropout, binary_out=False):

        self.max_seq_length = max_seq_length
        self.positional_embedding = nn.Embedding(
                num_embeddings=max_seq_length+1, embedding_dim=latent_dim)

        self.question_embedding = None  # GPT Encoder (text -> d/3)
        self.question_embedding2 = None  # GPT Encoder 2 (text -> d)
        self.response_embedding = nn.Embedding(num_embeddings=num_apis,
                                               embedding_dim=(latent_dim)//3)
        self.answer_embedding = nn.Embedding(num_embeddings=num_apis,
                                             embedding_dim=latent_dim//3)

        self.matn = nn.MultiheadAttention(
                embed_dim=latent_dim, num_heads=num_heads, dropout=dropout)

        self.ff1 = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.ff2 = nn.Linear(in_features=latent_dim, out_features=latent_dim)

        self.prediction = nn.Linear(in_features=latent_dim,
                                    out_features=max_seq_length)

        self.layernorm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, question_sequence,
                response_sequence, answer_sequence, is_embedded=False):

        if not is_embedded:
            question_sequence = self.question_embedding(question_sequence)
            question_sequence2 = self.question_embedding2(question_sequence)
            response_sequence = self.response_embedding(response_sequence)
        answer_sequence = self.response_embedding(answer_sequence)

        query = question_sequence2

        x = torch.cat(question_sequence, response_sequence,
                      answer_sequence, dim=-1)
        x += self.positional_embedding(torch.arange(len(x)).unsqueeze(-1))

        x = self.dropout(x)
        future_mask = None
        res = self.matn(query=query, key=x, value=x, future_masks=future_mask)
        out = self.dropout(self.ff2(self.dropout(F.relu(self.ff1(res)))))

        out += self.layernorm(res)
        out = self.layernorm(out)

        logits = self.prediction(out)
        return logits
