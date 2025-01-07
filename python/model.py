import torch
from torch import nn
import torch.nn.functional as F

### Cnn Module

class Conv2DBlock(nn.Module):
    def __init__(self, n_features):
        super(Conv2DBlock, self).__init__()
        # Top branch: single 2D conv -> BatchNorm -> ReLU -> Dropout
        self.top_branch = nn.Sequential(
            nn.Conv2d(in_channels=n_features, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.bot_branch = nn.Sequential(
            nn.Conv2d(n_features, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=160, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        top = self.top_branch(x)
        bottom = self.bot_branch(x)
        return top + bottom

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        )
    def forward(self, x):
        return self.block(x)

class LinearBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(LinearBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.3),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.3),

            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(p=0.3)
        )

    def forward(self, x):
        return self.block(x)

class CNNModule(nn.Module):
    def __init__(self, n_features, n_classes):
        super(CNNModule, self).__init__()

        #2d block
        self.conv2d = Conv2DBlock(n_features)
        #1d block
        self.conv1d = Conv1DBlock(128, 96)
        #linear block
        self.linear = LinearBlock(96, 96, 128)

    def forward(self, x):
        batch_size, n_frames, n_features, h, w = x.shape
        frame_embeddings = []

        for i in range(n_frames):
          frame = x[:, i, :, :, :]
          out = self.conv2d(frame)
          avg_pool1 = torch.squeeze(nn.functional.avg_pool2d(out, kernel_size=(1, 11)), dim=-1) #11 offensive players
          max_pool1 = torch.squeeze(nn.functional.max_pool2d(out, kernel_size=(1, 11)), dim=-1) #11 offensive players

          out = 0.7 * avg_pool1 + 0.3 * max_pool1

          out = self.conv1d(out)

          avg_pool2 = torch.squeeze(nn.functional.avg_pool1d(out, kernel_size=11), dim=-1)
          max_pool2 = torch.squeeze(nn.functional.max_pool1d(out, kernel_size=11), dim=-1)

          out = 0.7 * avg_pool2 + 0.3 * max_pool2
          out = self.linear(out)
          frame_embeddings.append(out)

        return torch.stack(frame_embeddings, dim=1)  #final output is frame embedding

### Attention Module

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=16):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension not divisible by n_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.scaling = self.head_dim ** -0.5

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def create_attention_mask(self, n_frames, device):
        attn_mask = torch.triu(torch.ones(n_frames, n_frames), diagonal=1).bool()
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        return attn_mask.to(device)

    def forward(self, x):
        batch_size, n_frames, embed_dim = x.shape

        # Linear projections and reshape for multi-head attention
        q = self.q_proj(x).reshape(batch_size, n_frames, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, n_frames, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, n_frames, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Apply the causal mask to the attention weights
        attn_mask = self.create_attention_mask(n_frames, x.device)
        attn_weights = attn_weights.masked_fill(attn_mask, -1e9)

        # Apply softmax to get attention probabilities
        # attn_weights = F.softmax(attn_weights, dim=-1)
        # Apply sigmoid to get attention probabilities
        attn_weights = torch.sigmoid(attn_weights)
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, n_frames, embed_dim)
        output = self.out_proj(attn_output)

        return output

class SelfAttentionModule(nn.Module):
    def __init__(self, embed_dim=128, num_heads=16, n_classes=2):
        super(SelfAttentionModule, self).__init__()
        self.multihead_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

        self.final_linear = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        attn_output = self.multihead_attn(x)
        attn_output = self.norm(x + attn_output)

        attn_output = attn_output.transpose(1, 2)


        play_embedding = F.avg_pool1d(attn_output, kernel_size=attn_output.size(-1))
        play_embedding = play_embedding.squeeze(-1)

        logits = self.final_linear(play_embedding)

        return logits

### Full Module

class FullModel(nn.Module):
    def __init__(self, n_features, n_classes, embed_dim = 128, num_heads=16):
        super(FullModel, self).__init__()
        self.cnn = CNNModule(n_features, n_classes)
        self.attention = SelfAttentionModule(embed_dim, num_heads, n_classes)

    def forward(self, x):
          n_frames = x.shape[1]
          frame_embeddings = self.cnn(x)

          preds = []
          for seq_len in range(1, n_frames+1):
              current_stack = frame_embeddings[:, :seq_len, :]
              current_logits = self.attention(current_stack)
              preds.append(current_logits)

          all_preds = torch.stack(preds, dim=1)

          return all_preds