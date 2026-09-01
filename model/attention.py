import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 0. Instantiate the linear layers in the following order: Key, Query, Value.
# 1. Biases are not used in Attention, so for all 3 nn.Linear() instances, pass in bias=False.
# 2. torch.transpose(tensor, 1, 2) returns a B x T x A tensor as a B x A x T tensor.
# 3. This function is useful:
#    https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
# 4. Apply the masking to the TxT scores BEFORE calling softmax() so that the future
#    tokens don't get factored in at all.
#    To do this, set the "future" indices to float('-inf') since e^(-infinity) is 0.
# 5. To implement masking, note that in PyTorch, tensor == 0 returns a same-shape tensor 
#    of booleans. Also look into utilizing torch.ones(), torch.tril(), and tensor.masked_fill(),
#    in that order.
class SingleHeadAttention(nn.Module):
    
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.K = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.V = nn.Linear(embedding_dim, attention_dim, bias=False)
    
    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        # Return your answer to 4 decimal places
        batch_size, context_length, embedding_dim = embedded.shape
        Q = self.Q(embedded)
        K = self.K(embedded)
        V = self.V(embedded)
        attention_dim_size = Q.shape[-1]

        attn_score = (Q @ K.transpose(1, 2)) / math.sqrt(attention_dim_size)
        mask = torch.tril(torch.ones(context_length, context_length))
        attn_score = attn_score.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_score, dim=-1)
        out = attn_weights @ V
        
        return torch.round(out, decimals=4)