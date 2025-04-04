import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    def __init__(self, num_users, num_items, 
                 embedding_size=64, mlp_layers=[128, 64, 32], dropout=0.3):
        super().__init__()
        
        # Embeddings
        self.user_gmf_emb = nn.Embedding(num_users, embedding_size)
        self.item_gmf_emb = nn.Embedding(num_items, embedding_size)
        self.user_mlp_emb = nn.Embedding(num_users, embedding_size)
        self.item_mlp_emb = nn.Embedding(num_items, embedding_size)
        
        # MLP
        mlp = []
        input_dim = embedding_size * 2
        for output_dim in mlp_layers:
            mlp.extend([
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = output_dim
        self.mlp = nn.Sequential(*mlp)
        
        # Output
        self.output = nn.Linear(embedding_size + mlp_layers[-1], 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for emb in [self.user_gmf_emb, self.item_gmf_emb, 
                   self.user_mlp_emb, self.item_mlp_emb]:
            nn.init.xavier_uniform_(emb.weight)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        
        nn.init.xavier_uniform_(self.output.weight)
    
    def forward(self, user, item):
        # GMF
        gmf_user = self.user_gmf_emb(user)
        gmf_item = self.item_gmf_emb(item)
        gmf = gmf_user * gmf_item
        
        # MLP
        mlp_user = self.user_mlp_emb(user)
        mlp_item = self.item_mlp_emb(item)
        mlp = self.mlp(torch.cat([mlp_user, mlp_item], dim=1))
        
        # Concatenate and predict
        pred = self.output(torch.cat([gmf, mlp], dim=1))
        return torch.sigmoid(pred).squeeze()
