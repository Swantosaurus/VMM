import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = torch.nn.MultiheadAttention(embed_dim=dim,
                                               num_heads=n_heads,
                                               dropout=dropout,
                                               batch_first=True)

    def forward(self, x):
        attn_output, attn_output_weights = self.att(x, x, x)
        return attn_output


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class VisualTransformer(nn.Module):
    def __init__(self, classes, layers=5, channels=3, heads=4, dropout=0.1, embeddingDimension=128, patchSize=8, imSize=128):
        self.channels = channels
        self.height = imSize
        self.width = imSize
        self.patchSize = patchSize

        super(VisualTransformer, self).__init__()
        self.embadding = PatchEmbedding(
            channels, patchSize, embeddingDimension)

        self.numPatches = (imSize // patchSize) ** 2
        self.posEmbedding = nn.Parameter(
            torch.rand(1, self.numPatches + 1, embeddingDimension)
        )
        self.clasToken = nn.Parameter(torch.rand(1, 1, embeddingDimension))

        self.layers = nn.ModuleList([])
        for i in range(layers):
            oneLayer = nn.Sequential(
                ResidualAdd(
                    nn.Sequential(
                        nn.LayerNorm(embeddingDimension),
                        Attention(embeddingDimension, heads, dropout)
                    )
                ),
                ResidualAdd(
                    nn.Sequential(
                        nn.LayerNorm(embeddingDimension),
                        nn.Linear(embeddingDimension,
                                  embeddingDimension * 4),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(embeddingDimension * 4,
                                  embeddingDimension),
                        nn.Dropout(dropout)
                    )
                )
            )

            self.layers.append(oneLayer)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embeddingDimension),
            nn.Linear(embeddingDimension, classes)
        )

    def forward(self, img):
        emb = self.embadding(img)

        batchSize, n, _ = emb.shape

        clasToken = repeat(self.clasToken, '1 1 d -> b 1 d', b=batchSize)
        emb = torch.cat([clasToken, emb], dim=1)
        emb += self.posEmbedding[:, :(n + 1)]

        for layer in self.layers:
            emb = layer(emb)

        return self.classifier(emb[:, 0, :])
