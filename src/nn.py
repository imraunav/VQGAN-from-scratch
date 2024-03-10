import torch
from torch import nn
from einops import rearrange
import math


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        if self.in_channels != self.out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        return self.residual(x) + self.block(x)


class Upsample(nn.Module):
    def __init__(self, channels, with_conv=False):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_channels = channels
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = 1 / math.sqrt(channels)

    def forward(self, x):
        h = self.norm(x)
        qkv = self.qkv(h)
        _, _, height, width = qkv.shape
        qkv = rearrange(qkv, "b c h w -> b c (h w)")

        q, k, v = torch.chunk(qkv, 3, dim=1)  # split across channel
        attn = torch.einsum("bct,bcs->bts", q, k) * self.scale
        attn = torch.softmax(attn, dim=2)

        a = torch.einsum("bts,bcs->bct", attn, v)
        return x * rearrange(a, "b c (h w) -> b c h w", h=height, w=width)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        z_dim,
        n_resblock=2,
        training_resolution=256,
        attention_resolution=[],
        ch=128,
        ch_mult=(1, 1, 1, 2, 2, 4),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.attention_resolution = attention_resolution
        m = len(ch_mult)
        channels = []
        for mult in ch_mult:
            channels.append(mult * ch)

        layers = [
            nn.Conv2d(
                in_channels,
                channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
            )
        ]

        resolution = training_resolution
        for i in range(m - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for _ in range(n_resblock):
                layers.append(ResnetBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attention_resolution:
                    layers.append(NonLocalBlock(out_channels))
            if i != (m - 2):
                layers.append(Downsample(in_channels, with_conv=True))
                resolution //= 2

        layers.append(ResnetBlock(in_channels, out_channels))
        layers.append(NonLocalBlock(out_channels))
        layers.append(ResnetBlock(out_channels, out_channels))
        layers.append(nn.GroupNorm(32, out_channels))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(out_channels, z_dim, kernel_size=3, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(
        self,
        z_dim,
        out_channels,
        n_resblock=2,
        training_resolution=16,
        attention_resolution=[],
        ch=128,
        ch_mult=(4, 2, 2, 1, 1),
    ):
        super().__init__()
        m = len(ch_mult)
        channels = []
        for mult in ch_mult:
            channels.append(mult * ch)
        img_channels = out_channels
        out_channels = channels[0]  # renaming this var
        layers = [
            nn.Conv2d(z_dim, out_channels, kernel_size=3, padding=1),
            ResnetBlock(out_channels, out_channels),
            NonLocalBlock(out_channels),
            ResnetBlock(out_channels, out_channels),
        ]
        resolution = training_resolution
        for i in range(m - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for _ in range(n_resblock):
                layers.append(ResnetBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attention_resolution:
                    layers.append(NonLocalBlock(out_channels))
                if i != 0:
                    layers.append(Upsample(in_channels, with_conv=True))
                    resolution *= 2
        layers.append(nn.GroupNorm(32, out_channels))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(out_channels, img_channels, kernel_size=3, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CodeBook(nn.Module):
    def __init__(self, num_code_vectors, z_dim, beta):
        super().__init__()
        self.num_code_vectors = num_code_vectors
        self.z_dim = z_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.num_code_vectors, self.z_dim)
        self.embedding.weight.data.uniform(
            -1 / self.num_code_vectors, 1 / self.num_code_vectors
        )

    def forward(self, z):
        z = rearrange(z, "b z h w -> b h w z")
        z_ = rearrange(z, "b h w z-> (b h w) z")

        dist = (
            torch.sum(z_**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1, keepdim=True)
            - 2 * torch.matmul(z_, self.embedding.weight.t())
        )  # kinda weird application of (a - b)^2
        min_encoding_indicies = torch.argmin(dist, dim=1)
        z_q = self.embedding(min_encoding_indicies).view(
            z.shape
        )  # break back to (b, h, w, z)

        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z.detach() - z_q) ** 2
        )

        z_q = z + (z_q - z).detach()  # allow backpropagation through quantized latents

        z_q = rearrange(z_q, "b h w z -> b z h w")

        return z_q, min_encoding_indicies, loss


class VQGAN(nn.Module):
    def __init__(
        self,
        in_channels,
        z_dim,
        out_channels,
        n_decoder_resblock=2,
        encoder_training_res=256,
        encoder_attn_res=(32, 16),
        en_ch=128,
        codebook_size=1024,
        n_decoder_resblock=2,
        decoder_training_res=256,
        decoder_attn_res=(32, 16),
        de_ch=128,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels,
            z_dim,
            n_encoder_resblock,
            encoder_training_res,
            encoder_attn_res,
            en_ch,
        )
        self.decoder = Decoder(
            z_dim,
            out_channels,
            n_decoder_resblock,
            decoder_training_res,
            decoder_attn_res,
            de_ch,
        )
        self.codebook = CodeBook(codebook_size, z_dim, beta)
        self.pre_quant_conv = nn.Conv2d(z_dim, z_dim, 1)
        self.post_quant_conv = nn.Conv2d(z_dim, z_dim, 1)

    def encode(self, imgs):
        z = self.encoder(imgs)
        quant_conv = self.pre_quant_conv(z)
        z_q, codebook_idx, q_loss = self.codebook(z)
        return z_q, codebook_idx, q_loss

    def decode(self, z):
        post_quant_conv = self.post_quant_conv(z)
        decoded_imgs = self.decoder(post_quant_conv)
        return decoded_imgs

    def forward(self, x):
        z_q, codebook_idx, q_loss = self.encode(x)
        decoded_imgs = self.decode(z_q)
        return decoded_imgs, codebook_idx, q_loss

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(
            perceptual_loss, last_layer_weight, retain_graph=True
        )[0]
        gan_loss_grads = torch.autograd.grad(
            gan_loss, last_layer_weight, retain_graph=True
        )[0]

        lam = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        lam = torch.clamp(lam, 0, 1e4).detach()
        return 0.8 * lam

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0):
        if i < threshold:
            disc_factor = value
        return value


# if __name__ == "__main__":
#     # en = Encoder(3, 3, attention_resolution=(32, 16))
#     de = Decoder(3, 3, attention_resolution=(16,))
#     # en.to('mps').half()
#     de.to('mps').half()
#     # inp = torch.randn((3, 3, 256, 256), dtype=torch.half, device='mps')
#     # z = en(inp)
#     # print(z.shape) # torch.Size([3, 3, 16, 16])
#     z = torch.randn((3, 3, 16, 16), dtype=torch.half, device='mps')
#     recon = de(z)
#     print(recon.shape)
