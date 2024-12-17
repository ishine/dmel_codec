from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from vector_quantize_pytorch import GroupedResidualFSQ

from dmel_codec.models.modules.firefly import ConvNeXtBlock


@dataclass
class FSQResult:
    z: torch.Tensor
    codes: torch.Tensor
    latents: torch.Tensor


class DownsampleFiniteScalarQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        n_groups: int = 1,
        levels: tuple[int] = (8, 5, 5, 5),  # Approximate 2**10
        downsample_factor: tuple[int] = (2, 2),
        downsample_dims: tuple[int] | None = None,
        is_dmel: bool = False
    ):
        super().__init__()

        if downsample_dims is None:
            downsample_dims = [input_dim for _ in range(len(downsample_factor))]
            
        self.is_dmel = is_dmel
        self.groups = n_groups
        all_dims = (input_dim,) + tuple(downsample_dims) if is_dmel == False else (input_dim // n_groups,) + tuple([dims // n_groups for dims in downsample_dims])

        self.residual_fsq = GroupedResidualFSQ(
            dim=input_dim,
            levels=levels,
            num_quantizers=n_codebooks,
            groups=n_groups,
        )

        self.downsample_factor = downsample_factor
        self.downsample_dims = downsample_dims

        self.downsample = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv1d(
                        all_dims[idx],
                        all_dims[idx + 1],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx + 1]),
                )
                for idx, factor in enumerate(downsample_factor)
            ]
        )

        self.upsample = nn.Sequential(
            *[
                nn.Sequential(
                    nn.ConvTranspose1d(
                        all_dims[idx + 1],
                        all_dims[idx],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx]),
                )
                for idx, factor in reversed(list(enumerate(downsample_factor)))
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, z) -> FSQResult:
        original_shape = z.shape
        
        z = self.downsample(z)
        
        if self.is_dmel:
            # z = rearrange(z, "(b g) f t -> b (g f) t", g = self.groups)
            z = z.contiguous().view(original_shape[0] // self.groups, self.groups * original_shape[1], -1)
        
        quantized, indices = self.residual_fsq(z.mT)
        result = FSQResult(
            z=quantized.mT,
            codes=indices.mT,
            latents=z,
        )
        
        if self.is_dmel:
            # result.z = rearrange(result.z, "b (g f) t -> (b g) f t", g = self.groups)
            result.z = result.z.contiguous().view(original_shape[0], original_shape[1], -1)
        
        result.z = self.upsample(result.z)
        
        if self.is_dmel:
            # result.z = rearrange(result.z, "(b g) f t -> b (g f) t", g = self.groups)
            result.z = result.z.contiguous().view(original_shape[0] // self.groups, self.groups * original_shape[1], -1)

        # Pad or crop z to match original shape
        diff = original_shape[-1] - result.z.shape[-1]
        left = diff // 2
        right = diff - left

        if diff > 0:
            result.z = F.pad(result.z, (left, right))
        elif diff < 0:
            result.z = result.z[..., left:-right]

        return result

    def encode(self, z):
        z = self.downsample(z)
        
        if self.is_dmel:
            z = rearrange(z, "(b g) f t -> b (g f) t", g = self.groups)
            
        _, indices = self.residual_fsq(z.mT)
        
        indices = rearrange(indices, "g b l r -> b (g r) l")
        return indices

    def decode(self, indices: torch.Tensor):
        indices = rearrange(indices, "b (g r) l -> g b l r", g=self.residual_fsq.groups)
        z_q = self.residual_fsq.get_output_from_indices(indices)
        
        if self.is_dmel:
            z_q = rearrange(z_q.mT, "b (g f) t -> (b g) f t", g = self.groups)
        
        z_q = self.upsample(z_q)
        
        if self.is_dmel:
            z_q = rearrange(z_q, "(b g) f t -> b (g f) t", g = self.groups)
            
        return z_q

    # def from_latents(self, latents: torch.Tensor):
    #     z_q, z_p, codes = super().from_latents(latents)
    #     z_q = self.upsample(z_q)
    #     return z_q, z_p, codes


if __name__ == "__main__":
    rvq = DownsampleFiniteScalarQuantize(
        n_codebooks=1,
        downsample_factor=(2, 2),
    )
    x = torch.randn(16, 512, 80)

    result = rvq(x)
    print(rvq)
    print(result.latents.shape, result.codes.shape, result.z.shape)

    # y = rvq.from_codes(result.codes)
    # print(y[0].shape)

    # y = rvq.from_latents(result.latents)
    # print(y[0].shape)
