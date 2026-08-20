"""Stage-B V2: blind eight-bit content-conditioned natural-image channel."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

ACTIVE_BIT_NAMES = tuple(f"regional_symbol_bit_{i}" for i in range(8))


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, activation: str = "silu") -> None:
        super().__init__()
        act = nn.SiLU if activation == "silu" else nn.GELU
        groups = max(1, min(8, cout // 4))
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(groups, cout), act(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(groups, cout), act())
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class EightBitCarrier(nn.Module):
    """Eight independent learned spatial bases; inputs are canonical [B,8] bits."""
    def __init__(self, size: int = 64) -> None:
        super().__init__()
        self.size = int(size)
        bases = torch.randn(8, 1, size, size)
        bases -= bases.mean((-2, -1), keepdim=True)
        bases /= bases.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        self.bases = nn.Parameter(bases)

    def normalized_bases(self) -> torch.Tensor:
        x = self.bases - self.bases.mean((-2, -1), keepdim=True)
        return x / x.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        if bits.ndim != 2 or bits.shape[1] != 8:
            raise ValueError("Stage-B V2 payload must have shape [B,8]")
        signed = bits.to(self.bases.dtype).mul(2).sub(1)
        return signed[:, :, None, None] * self.normalized_bases()[:, 0][None]

    def correlation_matrix(self) -> torch.Tensor:
        x = F.normalize(self.normalized_bases().flatten(1), dim=1)
        return x @ x.T


class ContentConditionedResidualEncoder(nn.Module):
    def __init__(self, image_size: int = 64, width: int = 16, activation: str = "silu",
                 mask_floor: float = .25) -> None:
        super().__init__(); self.image_size = image_size; self.mask_floor = float(mask_floor)
        self.carrier = EightBitCarrier(image_size)
        self.image_stem = ConvBlock(3, width, activation)
        self.payload_stem = ConvBlock(8, width, activation)
        self.fuse1 = ConvBlock(width * 2, width, activation)
        self.down_image = nn.Conv2d(width, width * 2, 3, 2, 1)
        self.down_payload = nn.Conv2d(width, width * 2, 3, 2, 1)
        self.fuse2 = ConvBlock(width * 4, width * 2, activation)
        self.bottleneck = ConvBlock(width * 2 + 8, width * 2, activation)
        self.up = nn.ConvTranspose2d(width * 2, width, 2, 2)
        self.decode = ConvBlock(width * 2, width, activation)
        self.residual_head = nn.Conv2d(width, 3, 1)
        self.carrier_skip = nn.Conv2d(8, 3, 1, bias=False)
        self.mask_head = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.residual_head.weight); nn.init.zeros_(self.residual_head.bias)
        nn.init.constant_(self.carrier_skip.weight, 1 / math.sqrt(8))
        nn.init.zeros_(self.mask_head.weight); nn.init.constant_(self.mask_head.bias, 3.0)

    def forward(self, image: torch.Tensor, bits: torch.Tensor, amplitude: float) -> dict[str, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B,3,H,W]")
        carrier = self.carrier(bits)
        if carrier.shape[-2:] != image.shape[-2:]:
            carrier = F.interpolate(carrier, image.shape[-2:], mode="bilinear", align_corners=False)
        xi, xp = self.image_stem(image), self.payload_stem(carrier)
        skip = self.fuse1(torch.cat((xi, xp), 1))
        low_carrier = F.avg_pool2d(carrier, 2)
        low = self.fuse2(torch.cat((self.down_image(skip), self.down_payload(xp)), 1))
        low = self.bottleneck(torch.cat((low, low_carrier), 1))
        features = self.decode(torch.cat((self.up(low), skip), 1))
        bounded = torch.tanh(self.residual_head(features) + self.carrier_skip(carrier))
        mask = self.mask_floor + (1.0 - self.mask_floor) * torch.sigmoid(self.mask_head(features))
        residual = float(amplitude) * mask * bounded
        watermarked = (image + residual).clamp(0, 1)
        return {"watermarked_image": watermarked, "residual": watermarked-image,
                "preclamp_residual": residual, "bounded_residual": bounded,
                "strength_mask": mask, "carrier_features": carrier}


class FixedHighPass(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernels = torch.tensor([[[0,-1,0],[-1,4,-1],[0,-1,0]], [[0,0,0],[-1,0,1],[0,0,0]],
                                [[0,-1,0],[0,0,0],[0,1,0]], [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]], dtype=torch.float32)
        self.register_buffer("kernels", kernels[:, None])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        grey = image.mean(1, keepdim=True)
        responses = F.conv2d(grey, self.kernels, padding=1)
        blur = F.avg_pool2d(grey, 3, 1, 1)
        responses[:, 3:4] = grey - blur
        return responses


class BlindMultiScaleDecoder(nn.Module):
    def __init__(self, width: int = 16, activation: str = "silu", image_size: int = 64) -> None:
        super().__init__(); self.highpass = FixedHighPass()
        self.register_buffer("analytical_bases", analytical_carrier_bases(image_size))
        with torch.no_grad():
            basis_rgb = self.analytical_bases[:, None].expand(-1, 3, -1, -1)
            hp_bases = self.highpass(basis_rgb)
            hp_bases = hp_bases / hp_bases.square().mean((-2,-1), keepdim=True).sqrt().clamp_min(1e-6)
        self.register_buffer("analytical_hp_bases", hp_bases)
        self.rgb_stem = nn.Sequential(nn.Conv2d(3, width, 5, padding=2), nn.SiLU())
        self.hp_stem = nn.Sequential(nn.Conv2d(4, width, 3, padding=1), nn.SiLU())
        self.fuse = ConvBlock(width * 2, width * 2, activation)
        self.scale1 = ConvBlock(width * 2, width * 4, activation)
        self.scale2 = ConvBlock(width * 4, width * 4, activation)
        # Fixed payload-independent matched responses make the analytical
        # prerequisite observable; learned RGB/high-pass features remain intact.
        self.output = nn.Linear(width * 4 + 8, 8)

    def forward(self, questioned_image: torch.Tensor) -> torch.Tensor:
        if questioned_image.ndim != 4 or questioned_image.shape[1] != 3:
            raise ValueError("decoder accepts only questioned RGB image [B,3,H,W]")
        x = self.fuse(torch.cat((self.rgb_stem(questioned_image), self.hp_stem(self.highpass(questioned_image))), 1))
        x = self.scale1(F.avg_pool2d(x, 2)); x = self.scale2(F.avg_pool2d(x, 2))
        matched = self.analytical_matched_features(questioned_image)
        return self.output(torch.cat((x.mean((-2, -1)), matched), 1))

    def analytical_matched_features(self, questioned_image: torch.Tensor) -> torch.Tensor:
        """Payload-independent high-pass matched responses for Phase-1 observability."""
        hp_bases = self.analytical_hp_bases
        if hp_bases.shape[-2:] != questioned_image.shape[-2:]:
            hp_bases = F.interpolate(hp_bases, questioned_image.shape[-2:], mode="bilinear", align_corners=False)
            hp_bases = hp_bases / hp_bases.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
        hp = self.highpass(questioned_image)
        return 25.0 * torch.einsum("bchw,kchw->bk", hp, hp_bases) / (hp.shape[1]*hp.shape[2]*hp.shape[3])

    def forward_analytical(self, questioned_image: torch.Tensor) -> torch.Tensor:
        """Same blind output head with learned branch held at zero during Phase 1."""
        zeros = questioned_image.new_zeros((len(questioned_image), self.output.in_features-8))
        return self.output(torch.cat((zeros, self.analytical_matched_features(questioned_image)), 1))


class NaturalChannelV2(nn.Module):
    architecture_version = "natural_channel_v2.0"
    def __init__(self, image_size: int = 64, width: int = 16, activation: str = "silu", mask_floor: float = .25) -> None:
        super().__init__(); self.encoder = ContentConditionedResidualEncoder(image_size, width, activation, mask_floor)
        self.decoder = BlindMultiScaleDecoder(width, activation, image_size)

    def forward(self, image: torch.Tensor, bits: torch.Tensor, amplitude: float) -> dict[str, torch.Tensor]:
        out = self.encoder(image, bits, amplitude); out["logits"] = self.decoder(out["watermarked_image"]); return out


def analytical_carrier_bases(size: int | tuple[int, int], *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Eight deterministic, zero-mean, RMS-one, mutually orthogonal DCT bases."""
    h, w = (size, size) if isinstance(size, int) else size
    y = (torch.arange(h, device=device, dtype=dtype) + .5)[:, None]
    x = (torch.arange(w, device=device, dtype=dtype) + .5)[None, :]
    pairs = ((25,31),(31,25),(27,29),(29,27),(23,31),(31,23),(25,29),(29,25))
    bases = torch.stack([torch.cos(math.pi*k*x/w)*torch.cos(math.pi*l*y/h) for k,l in pairs])
    bases -= bases.mean((-2,-1),keepdim=True)
    return bases / bases.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)


def analytical_residual(bits: torch.Tensor, size: tuple[int, int], amplitude: float) -> torch.Tensor:
    """Deterministic, eight-band orthogonal warm-up signal independent of image identity."""
    carrier = analytical_carrier_bases(size, device=bits.device, dtype=torch.float32); signed = bits.float().mul(2).sub(1)
    signal = torch.einsum("bk,khw->bhw", signed, carrier).div(math.sqrt(8))
    return float(amplitude) * signal[:, None].expand(-1, 3, -1, -1).tanh()
