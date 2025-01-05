import jiwer
import numpy as np
import torch
import torchaudio

# from pyvisqol.pyvisqol import Visqol
from speechbrain.inference.speaker import EncoderClassifier
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio

def transform_text_list(text_list):
    """
    处理文本列表，消除大小写和标点符号的影响

    Args:
        text_list: 文本列表

    Returns:
        list: 处理后的文本列表
    """

    def clean_text(text):
        # 转小写
        text = text.lower()
        # 移除标点符号 (可以根据需要添加更多标点符号)
        for punct in ',.!?;:""\'"()[]{ }、，。！？；：""' "【】《》-":
            text = text.replace(punct, " ")
        # 移除多余空格
        text = " ".join(text.split())
        return text

    return [clean_text(text) for text in text_list]


def wer(gt_audio, rec_audio, gt_text, processor, model, sample_rate=24000):

    if sample_rate != 16000:
        import torchaudio

        resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(gt_audio.device)
        gt_audio = resampler(gt_audio.view(-1, gt_audio.shape[-1])).view(gt_audio.shape[0], gt_audio.shape[1], -1)
        rec_audio = resampler(rec_audio.view(-1, rec_audio.shape[-1])).view(rec_audio.shape[0], rec_audio.shape[1], -1)

    # gt audio
    input_features = processor(
        [audio.numpy().squeeze(0) for audio in gt_audio.cpu()],
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features
    predicted_ids = model.generate(input_features.to(gt_audio.device))
    gt_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    # rec audio
    input_features = processor(
        [audio.numpy().squeeze(0) for audio in rec_audio.cpu()],
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features
    predicted_ids = model.generate(input_features.to(rec_audio.device))
    rec_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    try:  # if no words are predicted
        gt_transcription_clean = transform_text_list(gt_transcription)
        rec_transcription_clean = transform_text_list(rec_transcription)
        gt_text_clean = transform_text_list(gt_text)
        wer_gt = jiwer.wer(gt_text_clean, gt_transcription_clean)
        wer_rec = jiwer.wer(gt_text_clean, rec_transcription_clean)
    except ValueError:
        wer_gt = None
        wer_rec = None

    return wer_gt, wer_rec


def calculate_f0_corr(gt_audio: torch.Tensor, rec_audio: torch.Tensor, sample_rate=24000):
    f0_gt = torchaudio.functional.detect_pitch_frequency(gt_audio, sample_rate=sample_rate)
    f0_rec = torchaudio.functional.detect_pitch_frequency(rec_audio, sample_rate=sample_rate)
    valid = (f0_gt > 0) & (f0_rec > 0)

    # f0_rmse = torch.sqrt(torch.mean((f0_gt[valid] - f0_rec[valid]) ** 2)).item()
    f0_corr = torch.corrcoef(torch.stack([f0_gt[valid], f0_rec[valid]]))[0, 1].item()

    return f0_corr


def calculate_si_snr(gt_audio: torch.Tensor, rec_audio: torch.Tensor, eps=1e-8):
    if gt_audio.dim() > 2:
        gt_audio_cal = gt_audio.clone()
        rec_audio_cal = rec_audio.clone()
        gt_audio_cal = gt_audio_cal.squeeze(1)
        rec_audio_cal = rec_audio_cal.squeeze(1)

    si_snr = scale_invariant_signal_noise_ratio(rec_audio_cal, gt_audio_cal)

    return si_snr


def calculate_ci_sdr(gt_audio: torch.Tensor, rec_audio: torch.Tensor):
    return calculate_si_snr(gt_audio, rec_audio)  # Simplified placeholder


def calculate_stoi(rec_audio: torch.Tensor, gt_audio: torch.Tensor, sample_rate=24000):
    stoi = ShortTimeObjectiveIntelligibility(sample_rate).to(gt_audio.device)
    return stoi(rec_audio, gt_audio).item()


def calculate_spk_sim(
    gt_audio: torch.Tensor,
    rec_audio: torch.Tensor,
    model: EncoderClassifier,
    sample_rate: int = 24000,
):
    if sample_rate != 16000:
        import torchaudio

        resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(gt_audio.device)
        gt_audio = resampler(gt_audio.view(-1, gt_audio.shape[-1])).view(gt_audio.shape[0], gt_audio.shape[1], -1)
        rec_audio = resampler(rec_audio.view(-1, rec_audio.shape[-1])).view(rec_audio.shape[0], rec_audio.shape[1], -1)

    gt_audio = gt_audio.squeeze(1)
    rec_audio = rec_audio.squeeze(1)

    gt_embedding = model.encode_batch(gt_audio)
    rec_embedding = model.encode_batch(rec_audio)

    cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    similarity = cosine_sim(gt_embedding, rec_embedding)

    return similarity.mean().item()


def compute_codebook_usage(all_codes: torch.Tensor, audio_mask: torch.Tensor | None):
    """
    all_codes: torch.tensor.shape = [B, codebooks, T]
    audio_mask: torch.tensor.shape = [B, T]
        if audio_mask is None, then codes_mask is all ones
    """
    if audio_mask is None:
        codes_mask = torch.ones(size=(all_codes.shape[0], all_codes.shape[2])).to(all_codes.device)
    else:
        codes_mask = torch.nn.functional.interpolate(audio_mask, size=(all_codes.shape[-1],), mode="nearest").squeeze(1)

    with torch.no_grad():
        entropy = []
        for codebook_id in range(all_codes.shape[1]):
            codes_ = all_codes[:, codebook_id, :]
            counts = torch.bincount(codes_[codes_mask == 1])
            counts = (counts / counts.sum()).clamp(1e-10)
            entropy.append(-(counts * counts.log()).sum().item() * np.log2(np.e))
        return entropy


def calculate_pesq(rec_audio: torch.Tensor, gt_audio: torch.Tensor, sample_rate=24000):
    pesq = PerceptualEvaluationSpeechQuality(16000, "wb")

    # PESQ要求采样率为16k或8k
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(gt_audio.device)
        gt_audio = resampler(gt_audio)
        rec_audio = resampler(rec_audio)

    gt_audio_cal = gt_audio.clone()
    rec_audio_cal = rec_audio.clone()
    assert gt_audio_cal.shape == rec_audio_cal.shape

    if gt_audio_cal.dim() == 2:
        gt_audio_cal = gt_audio_cal.view(-1)
        rec_audio_cal = rec_audio_cal.view(-1)
        return pesq(rec_audio_cal, gt_audio_cal)

    elif gt_audio_cal.dim() == 3:
        gt_audio_cal = gt_audio_cal.view(gt_audio_cal.shape[0], -1)
        rec_audio_cal = rec_audio_cal.view(rec_audio_cal.shape[0], -1)
        pesq_list = []
        for i in range(gt_audio_cal.shape[0]):
            try:
                pesq_list.append(pesq(rec_audio_cal[i], gt_audio_cal[i]))
            except Exception as e:
                print(f"gt_audio.shape = {gt_audio_cal.shape}, rec_audio.shape = {rec_audio_cal.shape}, error = {e}")
        return np.mean(np.array(pesq_list))

    elif gt_audio_cal.dim() == 1:
        return pesq(rec_audio_cal, gt_audio_cal)

    else:
        raise ValueError("gt_audio dim must be 1, 2 or 3")


# def get_batch_visqol_similarity(
#     gt_audio: torch.Tensor | np.ndarray,
#     rec_audio: torch.Tensor | np.ndarray,
#     visqol: Visqol, sample_rate: int = 24000
# ):
#     if isinstance(gt_audio, torch.Tensor):
#         gt_audio = gt_audio.cpu().numpy().astype(np.float64)
#     if isinstance(rec_audio, torch.Tensor):
#         rec_audio = rec_audio.cpu().numpy().astype(np.float64)

#     assert gt_audio.shape == rec_audio.shape
#     if gt_audio.ndim > 2:
#         gt_audio = gt_audio.reshape(gt_audio.shape[0], -1)
#         rec_audio = rec_audio.reshape(rec_audio.shape[0], -1)

#     similarity_list = []
#     for i in range(gt_audio.shape[0]):
#         similarity = visqol.api.Measure(gt_audio[i], rec_audio[i])
#         similarity_list.append(similarity.moslqo)
#     return np.array(similarity_list).mean()


if __name__ == "__main__":
    import librosa

    gt_audio, _ = librosa.load(
        "/sdb/data1/speech/24kHz/LibriTTS/train-other-500/1569/141081/1569_141081_000029_000003.wav",
        sr=24000,
        mono=True,
        duration=4,
    )
    rec_audio, _ = librosa.load(
        "???",
        sr=24000,
        mono=True,
        duration=4,
    )
    gt_audio = torch.tensor(gt_audio).unsqueeze(0).unsqueeze(0)
    rec_audio = torch.tensor(rec_audio).unsqueeze(0).unsqueeze(0)
    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-resnet-voxceleb")
    similarity = calculate_spk_sim(gt_audio, rec_audio, model, sample_rate=24000)
    print(similarity)
