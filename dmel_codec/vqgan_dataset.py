import math

# import librosa as rosa
import random
# import time
# import typing as tp
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
# import torchaudio
import soundfile as sf

# from glob import glob
import torch
from lightning import LightningDataModule
from pedalboard.io import AudioFile

# from os import path
from torch.utils.data import DataLoader, Dataset  # , WeightedRandomSampler
from torchaudio import transforms as T
from utils.logger import RankedLogger

# import os


# from scipy.signal import sosfiltfilt
# from scipy.signal import butter, cheby1, cheby2, ellip, bessel
# from scipy.signal import resample_poly
# from moviepy.editor import VideoFileClip
# from pydub import AudioSegment
# import moviepy


# from torch_common import print_once
# from modification import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T

logger = RankedLogger(__name__, rank_zero_only=False)


class VQGANDataset(Dataset):

    def __init__(
        self,
        filelist: str | List[str],
        sample_rate: int,
        hop_length: int,
        slice_frames: Optional[int] = None,
    ):
        super().__init__()

        # if type(filelist) == str:
        # import pdb; pdb.set_trace()
        filelist = Path(filelist)
        # root = filelist.parent
        with open(filelist, "r") as f:
            self.files = [line.strip() for line in f.readlines() if line.strip()]

        # elif type(filelist) == list:
        #     self.files = []
        #     for file in filelist:
        #         file = Path(file)
        #         root = file.parent
        #         files = [
        #             root / line.strip()
        #             for line in file.read_text(encoding="utf-8").splitlines()
        #             if line.strip()
        #         ]
        #         self.files += files

        logger.info(f"the dataset sample {len(self.files)}")

        random.shuffle(self.files)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.slice_frames = slice_frames
        # import pdb; pdb.set_trace()
        self.sample_size = self.slice_frames * self.hop_length
        # 256 * 256 for training, 512 * 256 for inference
        self.target_sr = self.sample_rate # 24kHz

    def __len__(self):
        return len(self.files)

    def pad_crop_load(self, filename, randomize = True, extra_frames = 0, size_ratio = 0.9):

        # size_ratio 表示长度大于等于需求的size_ratio 倍的时候才考虑padding，否则直接跳过
        # randomize 表示是否需要随机选取中间的长度（默认True）
        # extra_frames 用于适配一些采样率转换带来的少数信号点的损失

        ext = filename.split(".")[-1]
        audio = None

        # audio_sr : 32000, 192000
        # audio_length : 0.3
        # self.target_sr : 48000
        # self.sample_size : 32768

        # try:
        if ext == "mp3":
            with AudioFile(filename) as f:
                audio_length = f.frames
                audio_sr = f.samplerate
                if audio_length * self.target_sr / audio_sr <= self.sample_size * size_ratio:
                    raise Exception(f"The audio file {filename} too short, skipped")

                required_length_frame = math.ceil(self.sample_size * audio_sr / self.target_sr) + extra_frames
                current_length_frame = audio_length
                max_ofs = max(0, current_length_frame - required_length_frame)

                offset = random.randint(0, max_ofs) if (randomize and max_ofs) else 0
                f.seek(offset)
                audio = f.read(min(required_length_frame, audio_length - offset))
                audio = torch.mean(torch.from_numpy(audio), dim=0, keepdim=True)

        else: # for wav, flac and other common waveform type
            info = sf.info(filename)
            audio_length = info.duration
            audio_sr = info.samplerate
            # check the length
            print(audio_length, self.target_sr, self.sample_size)
            if audio_length * self.target_sr <= self.sample_size * size_ratio or audio_length >= 1000:
                raise Exception(f"The audio file {filename} too short, skipped")

            # print(f'calced length : {audio_length * self.target_sr}, needed length : {self.sample_size}')

            required_length_frame = math.ceil(self.sample_size * audio_sr / self.target_sr) + extra_frames
            current_length_frame = math.floor(audio_length * audio_sr)
            max_ofs = max(0, current_length_frame - required_length_frame)

            # set random offset if self.randomize is true
            offset = random.randint(0, max_ofs) if (randomize and max_ofs) else 0
            # print(f'choosen offset : {offset}')
            # import soundfile as sf
            # sf.write('x_original_audio.wav', audio, audio_sr)
            # print(f'current_length_frame : {current_length_frame}, offset + required_length_frame : {offset + required_length_frame}')
            # print(f'start : {offset}, stop : {min(current_length_frame, offset + required_length_frame)}')
            audio, *_ = sf.read(filename, start=offset, stop=min(current_length_frame, offset + required_length_frame))
            # sf.write('x_original_audio_select.wav', audio, audio_sr)
            # print(f'shape audio : {audio.dim()}')
            if audio.ndim == 1:
                audio = torch.from_numpy(audio).to(torch.float32).unsqueeze(0)
            else:
                audio = torch.mean(torch.from_numpy(audio).to(torch.float32), dim=1, keepdim=True).squeeze(1).unsqueeze(0)
            # print(f'check audio shape : {au\dio.shape}')
            # import soundfile as sf
            # sf.write('x_original_audio_select.wav', audio, audio_sr)

        # except Exception as e: # for video type
        #     if ext == 'mp4':
        #         video = VideoFileClip(filename)
        #         audio = video.audio
        #         audio = audio.to_soundarray(fps=self.target_sr)
        #         if audio.ndim > 1:
        #             audio = audio.mean(axis=1)
        #         audio = torch.from_numpy(audio).to(torch.float32).unsqueeze(0)
        #         audio_length = audio.shape[-1]
        #         audio_sr = self.target_sr

        #         if audio_length * self.target_sr <= self.sample_size * size_ratio:
        #             raise Exception(f"The audio file {filename} too short, skipped")

        #         required_length_frame = self.sample_size
        #         current_length_frame = audio_length
        #         max_ofs = max(0, current_length_frame - required_length_frame)
        #         offset = random.randint(0, max_ofs) if (randomize and max_ofs) else 0

        #         audio = audio[:, offset : offset + self.sample_size]

        #     else:
        #         raise AssertionError

        if audio_sr != self.target_sr:
            resample_tf = T.Resample(audio_sr, self.target_sr)
            audio = resample_tf(audio)

        # print(f'check, audio.shape : {audio.shape[-1]}, self.sample_size : {self.sample_size}')
        if audio.shape[-1] < self.sample_size:
            tmp_audio = audio.new_zeros([1, self.sample_size])
            tmp_audio[:, :audio.shape[-1]] = audio[:, :]
            audio = tmp_audio
        else:
            audio = audio[:, :self.sample_size]

        audio /= torch.max(torch.abs(audio)) # required by NU-Wave2
        # print(audio.max(), audio.min())
        audio = audio[0]
        # print(audio.shape)
        # import pdb; pdb.set_trace()
        return audio

    def get_item(self, idx):
        audio_filename = self.files[idx]
        # audio, _ = librosa.load(file, sr=self.sample_rate, mono=True)
        # audio = librosa.util.normalize(audio) * 0.95
        audio = self.pad_crop_load(audio_filename, randomize=True)
        # Slice audio and features
        # if (
        #     self.slice_frames is not None
        #     and audio.shape[0] > self.slice_frames * self.hop_length
        # ):
        #     start = np.random.randint(
        #         0, audio.shape[0] - self.slice_frames * self.hop_length
        #     )
        #     audio = audio[start : start + self.slice_frames * self.hop_length]

        if len(audio) == 0:
            return None

        return {
            "audio": torch.FloatTensor(audio),
            "audio_file": audio_filename
        }

    def __getitem__(self, idx):
        try:
            return self.get_item(idx)
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Error loading {self.files[idx]}: {e}")
            return None


@dataclass
class VQGANCollator:
    def __call__(self, batch):
        batch = [x for x in batch if x is not None]

        audio_lengths = torch.tensor([len(x["audio"]) for x in batch])
        audio_maxlen = audio_lengths.max()

        # Rounds up to nearest multiple of 2 (audio_lengths)
        audios = []
        for x in batch:
            audios.append(
                torch.nn.functional.pad(x["audio"], (0, audio_maxlen - len(x["audio"])))
            )

        files = []
        for x in batch:
            files.append(x['audio_file'])

        return {
            "audios": torch.FloatTensor(torch.stack(audios)),
            "audio_lengths": audio_lengths,
            "audio_files": files
        }


class VQGANDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: VQGANDataset,
        val_dataset: VQGANDataset,
        batch_size: int = 32,
        num_workers: int = 4,
        val_batch_size: Optional[int] = None,
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=VQGANCollator(),
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=VQGANCollator(),
            num_workers=self.num_workers,
            persistent_workers=True,
        )


# -------------------------------------------------parallel 加载数据--------------------------------------------------
class VQGANParallelDataset(Dataset):

    def __init__(
        self,
        filelist: str | List[str],
        sample_rate: int,
        hop_length: int,
        slice_frames: Optional[int],
        parallel_num: int = 4,
    ):
        super().__init__()

        if type(filelist) == str:
            filelist = Path(filelist)
            root = filelist.parent

            self.files = [
                root / line.strip()
                for line in filelist.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        elif type(filelist) == list:
            self.files = []
            for file in filelist:
                file = Path(file)
                root = file.parent
                files = [
                    root / line.strip()
                    for line in file.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.files += files

        random.shuffle(self.files)
        logger.info(f"the dataset sample {len(self.files)}")

        self.parallel_num = parallel_num
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.slice_frames = slice_frames

    def __len__(self):
        return len(self.files)

    def get_item(self, idx):
        file = self.files[idx]

        audio, _ = librosa.load(file, sr=self.sample_rate, mono=True)
        audio = librosa.util.normalize(audio) * 0.95

        # 分份数
        total_length = len(audio)
        min_length_per_slice = self.slice_frames * self.hop_length
        if total_length < min_length_per_slice * self.parallel_num:
            raise ValueError(f"Audio length is too short. Expected at least {min_length_per_slice * self.parallel_num} samples but got {total_length} samples.")

        # Calculate the length of each slice
        slice_length = total_length // self.parallel_num

        slices = []
        for i in range(self.parallel_num):
            start = i * slice_length
            end = start + slice_length
            slice_y = audio[start:end]

            # Slice audio and features
            start = np.random.randint(
                0, slice_y.shape[0] - self.slice_frames * self.hop_length
            )
            slices.append(slice_y[start : start + self.slice_frames * self.hop_length])

        if len(audio) == 0:
            return None

        return {
            "audio": slices,
            "audio_file": file
        }

    def __getitem__(self, idx):
        try:
            return self.get_item(idx)
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Error loading {self.files[idx]}: {e}")
            return None


@dataclass
class VQGANCollator_parallel:
    def __call__(self, batch):
        batch = [x for x in batch if x is not None]

        audio_list = []
        audio_file_list = []
        for audio_dict in batch:
            audios = audio_dict['audio']
            file = audio_dict['audio_file']
            if len(audio_list) == 0:
                audio_list = audios
            else:
                audio_list += audios

            for _ in range(len(audios)):
                audio_file_list.append(file)

        audio_lengths = torch.tensor([len(x) for x in audio_list])
        audio_maxlen = audio_lengths.max()

        # Rounds up to nearest multiple of 2 (audio_lengths)
        audios = []
        for x in audio_list:
            audios.append(
                torch.nn.functional.pad(torch.FloatTensor(x), (0, audio_maxlen - len(x)))
            )

        return {
            "audios": torch.stack(audios),
            "audio_lengths": audio_lengths,
            "audio_files": audio_file_list
        }


class VQGANDataModule_parallel(LightningDataModule):

    def __init__(
        self,
        train_dataset: VQGANParallelDataset,
        val_dataset: VQGANDataset,
        batch_size: int = 32,
        num_workers: int = 4,
        val_batch_size: Optional[int] = None,
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=VQGANCollator_parallel(),
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=VQGANCollator(),
            num_workers=self.num_workers,
            persistent_workers=True,
        )

if __name__ == "__main__":
    dataset = VQGANParallelDataset("data/LibriTTS_R/vq_train_filelist.txt")
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=VQGANCollator_parallel()
    )

    for batch in dataloader:
        print(batch["audios"].shape)
        print(batch["features"].shape)
        print(batch["audio_lengths"])
        print(batch["feature_lengths"])
        break
