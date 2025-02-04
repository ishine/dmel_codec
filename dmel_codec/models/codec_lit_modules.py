import itertools
import math
import logging
import sys
from pathlib import Path
from typing import Any, Callable
import lightning as L
import torch
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from matplotlib import pyplot as plt
from torch import nn
from dmel_codec.utils.spectrogram import LogMelSpectrogram
from dmel_codec.models.modules.bigvgan.bigvgan import BigVGAN
from dmel_codec.models.modules.discriminator import Discriminator
from dmel_codec.models.modules.dowmsample_fsq import DownsampleFiniteScalarQuantize
from dmel_codec.models.modules.wavenet import WaveNet
from dmel_codec.utils.utils import avg_with_mask, plot_mel, sequence_mask
from dmel_codec.utils.logger import RankedLogger


log = RankedLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


class VQGAN(L.LightningModule):

    def __init__(
        self,
        encoder: WaveNet,
        quantizer: DownsampleFiniteScalarQuantize,
        vocoder: BigVGAN,
        encode_mel_transform: LogMelSpectrogram,
        gt_mel_transform: LogMelSpectrogram,
        optimizer: Callable | None = None,
        lr_scheduler: Callable | None = None,
        discriminator: Discriminator | None = None,
        decoder: WaveNet | None = None,
        weight_adv: float = 1.0,
        weight_vq: float = 1.0,
        weight_mel: float = 1.0,
        sampling_rate: int = 44100,
        freeze_encoder: bool = False,
        dmel_groups: int = 0,
        quanlity_linear: int = 768,
        dtype: torch.dtype | str = "bfloat16",
        accumulate_grad: int = 1,
    ):
        super().__init__()
        # torch.bfloat16 for str "bfloat16"
        log.info(f"dtype: {dtype}")
        if isinstance(dtype, str):
            self.encode_dtype = getattr(torch, dtype)
        else:
            self.encode_dtype = dtype
        log.info(f"VQGAN Using dtype: {self.encode_dtype}")

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Modules
        self.encoder = encoder
        self.quantizer = quantizer

        log.info(f"Vocoder ckpt path: {vocoder.ckpt_path}") # if vocoder.ckpt_path is None, the model is for training llm
        if vocoder.ckpt_path and Path(vocoder.ckpt_path).exists():
            vocoder.load_state_dict(
                torch.load(vocoder.ckpt_path, map_location="cpu")["generator"],
                strict=True,
            )
            self.vocoder = vocoder.eval()
            # Freeze vocoder
            for param in self.vocoder.parameters():
                param.requires_grad = False

            self.decoder = decoder
            self.discriminator = discriminator
        else:
            self.vocoder = None
            self.decoder = None
            self.discriminator = None

        self.encode_mel_transform = encode_mel_transform
        self.gt_mel_transform = gt_mel_transform

        # A simple linear layer to project quality to condition channels
        self.quality_projection = nn.Linear(1, quanlity_linear)

        # Loss weights
        self.weight_adv = weight_adv
        self.weight_vq = weight_vq
        self.weight_mel = weight_mel

        # Other parameters
        self.sampling_rate = sampling_rate

        # Disable strict loading
        self.strict_loading = False

        # If encoder is frozen
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

            for param in self.quantizer.parameters():
                param.requires_grad = False

        self.automatic_optimization = False
        self.dmel_groups = dmel_groups
        self.accumulate_grad = accumulate_grad

    def on_save_checkpoint(self, checkpoint):
        # Do not save vocoder
        state_dict = checkpoint["state_dict"]
        for name in list(state_dict.keys()):
            if "vocoder" in name:
                state_dict.pop(name)

    def configure_optimizers(self):
        optimizer_generator = self.optimizer_builder(
            itertools.chain(
                self.encoder.parameters(),
                self.quantizer.parameters(),
                self.decoder.parameters(),
                self.quality_projection.parameters(),
            )
        )
        optimizer_discriminator = self.optimizer_builder(
            self.discriminator.parameters()
        )

        lr_scheduler_generator = self.lr_scheduler_builder(optimizer_generator)
        lr_scheduler_discriminator = self.lr_scheduler_builder(optimizer_discriminator)

        return (
            {
                "optimizer": optimizer_generator,
                "lr_scheduler": {
                    "scheduler": lr_scheduler_generator,
                    "interval": "step",
                    "name": "optimizer/generator",
                },
            },
            {
                "optimizer": optimizer_discriminator,
                "lr_scheduler": {
                    "scheduler": lr_scheduler_discriminator,
                    "interval": "step",
                    "name": "optimizer/discriminator",
                },
            },
        )

    def expand_mask(self, mask_matrix):
        return mask_matrix.repeat_interleave(self.dmel_groups, dim=0)

    def training_step(self, batch, batch_idx):
        optim_g, optim_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()

        audios, audio_lengths = batch["audios"], batch["audio_lengths"]

        # log.info(f"audio_paths: {batch['audio_paths']}")
        # audios: (batch_size, 1, audio_length)

        audios = audios.float()

        with torch.no_grad():
            encode_mels = self.encode_mel_transform(audios)
            gt_mels = self.gt_mel_transform(audios)
            quality = ((gt_mels.mean(-1) > -8).sum(-1) - 90) / 10
            quality = quality.unsqueeze(-1)

        mel_lengths = audio_lengths // self.gt_mel_transform.hop_length
        mel_masks = sequence_mask(mel_lengths, gt_mels.shape[2])
        mel_masks_float_conv = mel_masks[:, None, :].to(self.encode_dtype)
        gt_mels = gt_mels * mel_masks_float_conv

        # Encode
        if self.dmel_groups > 0:
            dMel_masks_float_conv = self.expand_mask(mel_masks_float_conv)
            # encoded_dMels = rearrange(encoded_mels, "b (g f) t -> (b g) f t", g=self.dmel_groups) # can not work

            batch_size, num_mels, time_size = encode_mels.shape
            encode_dMels = encode_mels.contiguous().view(batch_size * self.dmel_groups, num_mels // self.dmel_groups, time_size)

            encode_dMels = encode_dMels * dMel_masks_float_conv
            encoded_features = self.encoder(encode_dMels) * dMel_masks_float_conv

        else:
            encoded_mels = encoded_mels * mel_masks_float_conv
            encoded_features = self.encoder(encoded_mels) * mel_masks_float_conv

        # Quantize
        vq_result = self.quantizer(encoded_features)
        loss_vq = getattr(vq_result, "loss", 0.0)
        vq_recon_features = vq_result.z * mel_masks_float_conv
        vq_recon_features = (
            vq_recon_features + self.quality_projection(quality)[:, :, None]
        )

        # VQ Decode
        gen_mel = (
            self.decoder(
                torch.randn_like(vq_recon_features) * mel_masks_float_conv,
                condition=vq_recon_features * mel_masks_float_conv,
            )
            * mel_masks_float_conv
        )

        # Discriminator
        real_logits = self.discriminator(gt_mels)
        fake_logits = self.discriminator(gen_mel.detach())
        d_mask = F.interpolate(
            mel_masks_float_conv, size=(real_logits.shape[2],), mode="nearest"
        )

        loss_real = avg_with_mask((real_logits - 1) ** 2, d_mask)
        loss_fake = avg_with_mask(fake_logits**2, d_mask)

        loss_d = (loss_real + loss_fake) / self.accumulate_grad # align with learning rate

        self.log(
            "train/discriminator/loss",
            loss_d * self.accumulate_grad,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        # Discriminator backward
        self.manual_backward(loss_d)

        if (batch_idx + 1) % self.accumulate_grad == 0:
            self.clip_gradients(
                optim_d, gradient_clip_val=1000.0, gradient_clip_algorithm="norm"
            )
            optim_d.step()
            optim_d.zero_grad()
            scheduler_d.step()

        # Mel Loss, applying l1, using a weighted sum
        mel_distance = (
            gen_mel - gt_mels
        ).abs()  # * 0.5 + self.ssim(gen_mel, gt_mels) * 0.5
        loss_mel_low_freq = avg_with_mask(mel_distance[:, :40, :], mel_masks_float_conv)
        loss_mel_mid_freq = avg_with_mask(
            mel_distance[:, 40:70, :], mel_masks_float_conv
        )
        loss_mel_high_freq = avg_with_mask(
            mel_distance[:, 70:, :], mel_masks_float_conv
        )
        loss_mel_all_band = avg_with_mask(
            mel_distance, mel_masks_float_conv
        )

        loss_mel = (
            loss_mel_low_freq * 0.6 + loss_mel_mid_freq * 0.3 + loss_mel_high_freq * 0.1
        ) * 0.5 + loss_mel_all_band * 0.5

        # Adversarial Loss
        fake_logits = self.discriminator(gen_mel)
        loss_adv = avg_with_mask((fake_logits - 1) ** 2, d_mask)

        # Total loss
        loss = (
            self.weight_vq * loss_vq
            + self.weight_mel * loss_mel
            + self.weight_adv * loss_adv
        ) / self.accumulate_grad # align with learning rate

        # Log losses
        self.log(
            "train/generator/loss",
            loss * self.accumulate_grad,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "train/generator/loss_vq",
            loss_vq,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "train/generator/loss_mel",
            loss_mel,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "train/generator/loss_adv",
            loss_adv,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        # Generator backward
        self.manual_backward(loss)

        # for name, params in self.named_parameters():
        #     if params.requires_grad == True and params.grad == None:
        #         print(f'{name} has no grad')

        if (batch_idx + 1) % self.accumulate_grad == 0:
            self.clip_gradients(
                optim_g, gradient_clip_val=1000.0, gradient_clip_algorithm="norm"
            )
            optim_g.step()
            optim_g.zero_grad()
            scheduler_g.step()


    def validation_step(self, batch: Any, batch_idx: int):
        audios, audio_lengths = batch["audios"], batch["audio_lengths"]
        # audios: (batch_size, 1, audio_length)
        # log.info(f"audio_paths: {batch['audio_paths']}")
        audios = audios.float()
        batch_size = audios.shape[0]

        encode_mels = self.encode_mel_transform(audios)
        gt_mels = self.gt_mel_transform(audios)

        mel_lengths = audio_lengths // self.gt_mel_transform.hop_length
        mel_masks = sequence_mask(mel_lengths, gt_mels.shape[2])
        mel_masks_float_conv = mel_masks[:, None, :].to(self.encode_dtype)
        gt_mels = gt_mels * mel_masks_float_conv

        # Encode
        if self.dmel_groups > 0:
            dMel_masks_float_conv = self.expand_mask(mel_masks_float_conv)
            # encoded_dMels = rearrange(encode_mels, "b (g f) t -> (b g) f t", g=self.dmel_groups)

            batch_size, num_mels, time_size = encode_mels.shape
            encode_dMels = encode_mels.contiguous().view(batch_size * self.dmel_groups, num_mels // self.dmel_groups, time_size)

            encode_dMels = encode_dMels * dMel_masks_float_conv
            encoded_features = self.encoder(encode_dMels) * dMel_masks_float_conv

        else:
            encode_mels = encode_mels * mel_masks_float_conv
            encoded_features = self.encoder(encode_mels) * mel_masks_float_conv

        # Quantize
        vq_recon_features = self.quantizer(encoded_features).z * mel_masks_float_conv
        vq_recon_features = (
            vq_recon_features
            + self.quality_projection(
                torch.ones(
                    vq_recon_features.shape[0], 1, device=vq_recon_features.device
                )
                * 2
            )[:, :, None]
        )

        # VQ Decode
        gen_aux_mels = (
            self.decoder(
                torch.randn_like(vq_recon_features) * mel_masks_float_conv,
                condition=vq_recon_features,
            )
            * mel_masks_float_conv
        )
        loss_mel = avg_with_mask((gen_aux_mels - gt_mels).abs(), mel_masks_float_conv)

        self.log(
            "val_loss",
            loss_mel,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        if self.vocoder is None:
            raise ValueError("Vocoder is not loaded")
        recon_audios = self.vocoder(gt_mels)
        gen_aux_audios = self.vocoder(gen_aux_mels)

        # only log the first batch
        if batch_idx >= 4:
            return

        for idx, (
            gt_mel,
            gen_aux_mel,
            audio,
            gen_aux_audio,
            recon_audio,
            audio_len,
        ) in enumerate(
            zip(
                gt_mels,
                gen_aux_mels,
                audios.cpu().float(),
                gen_aux_audios.cpu().float(),
                recon_audios.cpu().float(),
                audio_lengths[0], # audio_lengths: (1, batch_size)
            )
        ):
            if idx > 0:
                break

            mel_len = audio_len // self.gt_mel_transform.hop_length

            image_mels = plot_mel(
                [
                    gt_mel[:, :mel_len],
                    gen_aux_mel[:, :mel_len],
                ],
                [
                    "Ground-Truth",
                    "Auxiliary",
                ],
            )

            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure(
                    f"sample-{batch_idx}-{idx}/mels",
                    image_mels,
                    global_step=self.global_step,
                )
                self.logger.experiment.add_audio(
                    f"sample-{batch_idx}-{idx}/wavs/gt",
                    audio[0, :audio_len],
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )
                self.logger.experiment.add_audio(
                    f"sample-{batch_idx}-{idx}/wavs/gen",
                    gen_aux_audio[0, :audio_len],
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )
                self.logger.experiment.add_audio(
                    f"sample-{batch_idx}-{idx}/wavs/recon",
                    recon_audio[0, :audio_len],
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )

            plt.close(image_mels)

    def encode(self, audios, audio_lengths): # return indices and indices_lengths
        encoded_features, mel_lengths = self.encode_unquantized(audios, audio_lengths)
        indices, indices_lengths = self.get_indices_from_unquantized_features(encoded_features, mel_lengths)

        return indices, indices_lengths

    def decode(self, indices, feature_lengths, return_audios=False): # return audios or mel
        z, mel_masks_float_conv = self.get_quantized_features_from_indices(indices, feature_lengths)

        gen_mel = (
            self.decoder(
                torch.randn_like(z).to(self.encode_dtype) * mel_masks_float_conv,
                condition=z,
            )
            * mel_masks_float_conv
        )

        if return_audios:
            if self.vocoder is None:
                raise ValueError("Vocoder is not loaded")
            return self.vocoder(gen_mel), gen_mel

        return gen_mel
    
    def encode_unquantized(self, audios, audio_lengths): # return unquantized_features and mel_lengths
        audios = audios.float()

        mels = self.encode_mel_transform(audios) # mel must be float32
        mels = mels.to(self.encode_dtype) 

        mel_lengths = audio_lengths // self.encode_mel_transform.hop_length

        mel_masks = sequence_mask(mel_lengths, mels.shape[2])
        mel_masks_float_conv = mel_masks[:, None, :].to(self.encode_dtype)

        # Encode
        if self.dmel_groups > 0:
            dMel_masks_float_conv = self.expand_mask(mel_masks_float_conv).to(self.encode_dtype).to(mels.device)
            # encoded_dMels = rearrange(
            #     mels, "b (g f) t -> (b g) f t", g=self.dmel_groups
            # )
            batch_size, num_mels, time_size = mels.shape
            encoded_dMels = mels.contiguous().view(batch_size * self.dmel_groups, num_mels // self.dmel_groups, time_size)

            encoded_dMels = encoded_dMels * dMel_masks_float_conv
            encoded_features = self.encoder(encoded_dMels) * dMel_masks_float_conv

        else:
            encoded_mels = mels * mel_masks_float_conv
            encoded_features = self.encoder(encoded_mels) * mel_masks_float_conv

        return encoded_features, mel_lengths
    
    def get_quantized_features_from_indices(self, indices, feature_lengths): # return quantized_features
        factor = math.prod(self.quantizer.downsample_factor)
        mel_masks = sequence_mask(feature_lengths * factor, indices.shape[2] * factor)
        mel_masks_float_conv = mel_masks[:, None, :].to(self.encode_dtype).to(indices.device)

        z = self.quantizer.decode(indices) * mel_masks_float_conv
        z = (
            z
            + self.quality_projection(torch.ones(z.shape[0], 1, device=z.device).to(self.encode_dtype) * 2)[
                :, :, None
            ]
        )
        return z, mel_masks_float_conv

    def get_indices_from_unquantized_features(self, unquantized_features, mel_lengths): # return indices
        indices_lengths = mel_lengths // math.prod(self.quantizer.downsample_factor)
        return self.quantizer.encode(unquantized_features), indices_lengths