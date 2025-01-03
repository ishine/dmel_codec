import glob
import os
import torch
from matplotlib import pyplot as plt
from dmel_codec.utils.logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def find_lastest_ckpt(directory):
    if directory is None:
        return None
    ckpt_file = glob.glob(os.path.join(directory, "*.ckpt"))

    if not ckpt_file:
        log.info(f"No ckpt files found in this directory: {directory}")
        return None

    latest_ckpt_file = max(ckpt_file, key=os.path.getmtime)
    return latest_ckpt_file

def plot_mel(data, titles=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)

    if titles is None:
        titles = [None for i in range(len(data))]

    plt.tight_layout()

    for i in range(len(data)):
        mel = data[i]

        if isinstance(mel, torch.Tensor):
            mel = mel.float().detach().cpu().numpy()

        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    return fig

def sequence_mask(length: torch.Tensor, max_length: int | None = None):
    # length: (batch_size, ) or (1, batch_size)
    if length.ndim == 2:
        length = length.squeeze(0)
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def avg_with_mask(x, mask):
    assert mask.dtype == torch.float, "Mask should be float"

    if mask.ndim == 2:
        mask = mask.unsqueeze(1)

    if mask.shape[1] == 1:
        mask = mask.expand_as(x)

    return (x * mask).sum() / mask.sum()

def open_filelist(filelist_path, file_num=None):
    audio_path_list = []
    with open(filelist_path, 'r') as f:
        filelist = f.readlines()
        if file_num is None:
            file_num = len(filelist)
        for file in filelist[:file_num]:
            audio_path = file.strip()
            audio_path_list.append(audio_path)
    return audio_path_list