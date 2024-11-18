class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
class VisionTransformerDecoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, h_resolution: int, w_resolution: int, patch_size: int, stride_size: int, output_dim: int):

        super().__init__()
        self.h_resolution = h_resolution
        self.w_resolution = w_resolution
        self.width = width
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.output_dim = output_dim

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=width,
            out_channels=output_dim,
            kernel_size=patch_size,
            stride=stride_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor):  #torch.Size([32, 129, 768])

        # Transformer processing  
        x = x.permute(1, 0, 2)  # NLD -> LND  #torch.Size([129, 32, 768])
        x = self.transformer(x)  # Transformer Decoder   #torch.Size([129, 32, 768])
        x = x.permute(1, 0, 2)  # LND -> NLD  #torch.Size([128, 129, 768])

        # LayerNorm
        x = self.ln_post(x) ###torch.Size([32, 129, 768])

        # Reshape into image-like shape
        batch_size = x.shape[0]
        grid_size = self.h_resolution * self.w_resolution
        x = x[:, 1:, :]  # Exclude class token
        x = x.transpose(1, 2)  # NLD -> NCL #torch.Size([32, 768, 128])
        x = x.reshape(batch_size, self.width, self.h_resolution, self.w_resolution)  #torch.Size([32, 768, 16, 8])

        # Transposed convolution to reconstruct the image
        x_reconstructed = self.conv_transpose(x) #torch.Size([32, 512, 256, 128]) -> #torch.Size([32, 3, 256, 128])

        return x_reconstructed


###################### use #########################
        self.decoder = VisionTransformerDecoder(    
            width=768, 
            layers=12, 
            heads=12, 
            h_resolution=16, 
            w_resolution=8, 
            patch_size=16, 
            stride_size=16, 
            output_dim=3
        )
