import pandas as pd
from model import SAOKT
import torch

sequence_length = 50
qdim = 128
num_heads = 2
num_apis = 50
dropout = 0.02

data_file = './small_dataset.pkl'

# Step 1: read data
# Step 2: create model
# Step 3: train model
# Step 4: evaluate model


m = SAOKT(qdim=qdim, max_seq_length=sequence_length,
          num_heads=num_heads, num_apis=num_apis, dropout=dropout)

dataset = pd.read_pickle(data_file)
question_sequence = torch.randn(size=(sequence_length, qdim))
response_sequence = torch.randn(size=(sequence_length, qdim))
answer_sequence = torch.zeros(sequence_length).long()

out = m.forward(question_sequence=question_sequence, response_sequence=response_sequence, answer_sequence=answer_sequence)
print(out.shape)

