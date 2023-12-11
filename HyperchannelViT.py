import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

class LCFE_Fusion(nn.Module):
    # fusing the extracted features with the original ones
    def __init__(self, near_band=5, img_size=7, reduction=1):
        super().__init__()

        self.img_size = img_size
        self.LCFE = LCFE(channel=near_band, cube_size=img_size, reduction=reduction)
        self.w = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x1 = self.LCFE(x)
        x1 = self.w * x[:,:,0,:,:].unsqueeze(2) + x1

        return x1


class LCFE(nn.Module):
    # Parallel operation of grouped bands
    def __init__(self, channel, cube_size=7, reduction=16):
        super().__init__()

        # Each band group shares the convolutional kernel of pooling
        self.maxpool = nn.AdaptiveMaxPool2d(cube_size)
        self.avgpool = nn.AdaptiveAvgPool2d(cube_size)
        # Mlp for sharing and integrating spectral information within groups
        # (using 1D convolution to achieve fully connected layers)
        self.interaction = nn.Sequential(
            nn.Conv2d(channel, channel // 2, 1, bias=False),  # in_channel,out_channel,kernel
            nn.ReLU(),
            nn.Conv2d(channel // 2, 1, 1, bias=False)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Merge all groups within the batch into the first dimension (Restore it when output)
        b,p,n,i,j = x.shape
        x_reshape = torch.reshape(x,(b*p,n,i,j))
        del x

        max_result=self.maxpool(x_reshape)
        avg_result=self.avgpool(x_reshape)
        max_out=self.interaction(max_result)
        avg_out=self.interaction(avg_result)
        del max_result
        del avg_result
        output=self.sigmoid(max_out+avg_out)
        del max_out
        del avg_out

        return torch.reshape(output, (b, p, 1, i, j))  # The grouping dimension has been integrated into 1 when output


class hyperchannelViT(nn.Module):
    def __init__(self,
                 patch_size,  # args.patches, spatial size of patch
                 near_band,  # the number of bands in a group
                 band,  # the number of groups, equal to the number of bands in a hyperspectral image (carrying neighboring bands one by one)
                 num_classes,
                 dim,  # the dimension when put into the encoders
                 depth,  # the stack number of encoders
                 heads,  # the number of heads in multi-head attention
                 mlp_dim,
                 dim_head=16,
                 dropout=0.,
                 emb_dropout=0.,
                 mode='LCFE',
                 backbone='ViT',
                 ):
        super().__init__()

        if mode == 'LCFE':
            patch_dim = patch_size ** 2  # the grouping dimension will be integrated into 1 when output
        else:
            patch_dim = patch_size ** 2 * near_band  # the flatten dimension carrying group num
        self.near_band = near_band
        self.mode = mode
        self.backbone = backbone
        if mode == 'LCFE':
            self.LCFE_X = LCFE_Fusion(near_band=near_band, img_size=patch_size, reduction=1)
        # Initialize cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # liner projection by a learnable matrix
        self.patch_to_embedding = nn.Linear(patch_dim, dim)

        self.dropout = nn.Dropout(emb_dropout)

        self.pos_embedding = nn.Parameter(torch.randn(1, band + 1, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, band, backbone)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, mask=None):
        if self.mode == 'LCFE':
            x = self.LCFE_X(torch.as_tensor(x).permute(0, 1, 4, 2, 3))  # (batch, patch_num, patch_size, patch_size, near_band)
            # flatten
            x = torch.reshape(x,[x.shape[0],x.shape[1],x.shape[2]*x.shape[3]*x.shape[4]])
        # linear projection
        x = self.patch_to_embedding(x)  # [b,n,dim]
        b, n, _ = x.shape

        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # [b,1,dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [b,n+1,dim]
        x += self.pos_embedding[:, :(n + 2)]
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x, mask)

        # classification: using cls_token output
        x = self.to_latent(x[:, 0])

        # MLP classification layer

        return self.mlp_head(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))

    def forward(self, x, mask=None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask=mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 1:
                    x = self.skipcat[nl - 2](
                        torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, mask=mask)
                x = ff(x)
                nl += 1

        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out