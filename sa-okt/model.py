# layers needed:
# 1: embedding layer
# 2: multihead attention
# 3: dropout
# 4: normalization
# 5: FFN (repeat 3, 4)

# data flow

# 1: input interaction sequence I: (N, x, d)
#       N is # interactions in a sequence
#       x is max length of an interaction
#       d is embedding dimension
#
#       each interaction consists of
#       (question, response, real_answer, correctness)

# 2: feed emedded sequence I through attention layers
#       I -> E
#       might be hard to do this, each element of
#       each sequence consists of multiple tensors
#
#       we have to connect x & d in such a way
#       (concat question embds, response, etc) into Q and K
#       or just concat the entire interaction into one tensor

# 3: feed E through FNN
#       dropout + regularization
#       train using cross-entropy
