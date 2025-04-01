import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128, mlp_dims=[256, 128, 64], dropout=0.2):
        super(NCF, self).__init__()

        # Embedding layers
        self.user_embedding_GMF = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_GMF = nn.Embedding(num_items, embedding_dim)

        self.user_embedding_MLP = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_MLP = nn.Embedding(num_items, embedding_dim)

        # MLP layers
        mlp_input_size = embedding_dim * 2
        mlp_layers = []
        for dim in mlp_dims:
            mlp_layers.append(nn.Linear(mlp_input_size, dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            mlp_input_size = dim
        self.mlp = nn.Sequential(*mlp_layers)

        # Final layer (GMF + MLP combined)
        fusion_dim = embedding_dim + mlp_dims[-1]
        self.output_layer = nn.Linear(fusion_dim, 1)

    def forward(self, user_indices, item_indices):
        # GMF
        gmf_user = self.user_embedding_GMF(user_indices)
        gmf_item = self.item_embedding_GMF(item_indices)
        gmf_output = gmf_user * gmf_item 

        # MLP
        mlp_user = self.user_embedding_MLP(user_indices)
        mlp_item = self.item_embedding_MLP(item_indices)
        mlp_input = torch.cat((mlp_user, mlp_item), dim=1)
        mlp_output = self.mlp(mlp_input)

        # Concatenate GMF and MLP
        final_input = torch.cat((gmf_output, mlp_output), dim=1)

        # Prediction
        prediction = torch.sigmoid(self.output_layer(final_input)).squeeze()
        return prediction
