import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=64, mlp_layers=[128, 64, 32], dropout=0.2):
        super(NCF, self).__init__()
        # Embeddings for GMF and MLP
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_size)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_size)
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_size)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_size)

        # Build MLP layers with Dropout
        mlp_modules = []
        input_size = embedding_size * 2
        for output_size in mlp_layers:
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout))
            input_size = output_size
        self.mlp = nn.Sequential(*mlp_modules)

        # Fusion and output
        fusion_input_size = embedding_size + mlp_layers[-1]
        self.output_layer = nn.Linear(fusion_input_size, 1)

    def forward(self, user, item):
        # GMF path
        gmf_user = self.user_embedding_gmf(user)
        gmf_item = self.item_embedding_gmf(item)
        gmf_output = gmf_user * gmf_item

        # MLP path
        mlp_user = self.user_embedding_mlp(user)
        mlp_item = self.item_embedding_mlp(item)
        mlp_input = torch.cat((mlp_user, mlp_item), dim=-1)
        mlp_output = self.mlp(mlp_input)

        # Combine GMF and MLP
        fusion_input = torch.cat((gmf_output, mlp_output), dim=-1)
        output = self.output_layer(fusion_input)
        return torch.sigmoid(output).squeeze()
