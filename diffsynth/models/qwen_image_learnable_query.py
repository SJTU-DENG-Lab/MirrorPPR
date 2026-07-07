import torch
import torch.nn as nn

class QwenImageLearnableQuery(nn.Module):
    def __init__(self, query_length: int = 64, hidden_state_dim: int = 3584, initializer_range: float = 0.02):
        super().__init__()
        self.query_length = query_length
        self.hidden_state_dim = hidden_state_dim
        self.learnable_query = nn.Parameter(torch.randn(query_length, hidden_state_dim) * initializer_range)

    def forward(self, batch_size: int):
        return self.learnable_query.unsqueeze(0).expand(batch_size, -1, -1)

                                                 

              
                                         
                                 
                                                                                                             
