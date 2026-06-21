#!/usr/bin/env python3
"""Run AutoVC conversion for every row in the shared manifest.

The script adapts the upstream notebook workflow to the baseline output
contract used by the evaluator:

    voicegrad/baselines/autovc/generated_wavs/<src>_to_<tgt>/<utt_id>.wav
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from scipy.signal import get_window


SAMPLE_RATE = 16000
FFT_LENGTH = 1024
HOP_LENGTH = 256
N_MELS = 80
MEL_FMIN = 90
MEL_FMAX = 7600
MIN_LEVEL = np.exp(-100 / 20 * np.log(10))


def butter_highpass(cutoff: float, fs: int, order: int = 5) -> tuple[np.ndarray, np.ndarray]:
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    return signal.butter(order, normal_cutoff, btype="high", analog=False)


def py_stft(x: np.ndarray, fft_length: int = FFT_LENGTH, hop_length: int = HOP_LENGTH) -> np.ndarray:
    x = np.pad(x, int(fft_length // 2), mode="reflect")
    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    window = get_window("hann", fft_length, fftbins=True)
    return np.abs(np.fft.rfft(window * frames, n=fft_length).T)


def load_wav(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    import librosa

    wav, file_sr = sf.read(path, always_2d=False)
    wav = np.asarray(wav, dtype=np.float64)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    if wav.size == 0:
        raise ValueError(f"Empty wav file: {path}")
    return wav


def mel_basis() -> np.ndarray:
    import librosa

    try:
        return librosa.filters.mel(
            sr=SAMPLE_RATE,
            n_fft=FFT_LENGTH,
            fmin=MEL_FMIN,
            fmax=MEL_FMAX,
            n_mels=N_MELS,
        ).T
    except TypeError:
        return librosa.filters.mel(SAMPLE_RATE, FFT_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX, n_mels=N_MELS).T


def wav_to_autovc_mel(path: Path, basis: np.ndarray) -> np.ndarray:
    b, a = butter_highpass(30, SAMPLE_RATE, order=5)
    wav = load_wav(path, SAMPLE_RATE)
    filtered = signal.filtfilt(b, a, wav)
    spectrogram = py_stft(filtered).T
    mel = np.dot(spectrogram, basis)
    mel_db = 20 * np.log10(np.maximum(MIN_LEVEL, mel)) - 16
    normalized = np.clip((mel_db + 100) / 100, 0, 1)
    return normalized.astype(np.float32)


def pad_seq(x: np.ndarray, base: int = 32) -> tuple[np.ndarray, int]:
    len_out = int(base * math.ceil(float(x.shape[0]) / base))
    len_pad = len_out - x.shape[0]
    return np.pad(x, ((0, len_pad), (0, 0)), "constant"), len_pad


def clean_state_dict(state_dict, prefix: str = "module.") -> OrderedDict:
    cleaned: OrderedDict = OrderedDict()
    for key, value in state_dict.items():
        cleaned[key[len(prefix) :] if key.startswith(prefix) else key] = value
    return cleaned


def choose_device(requested: str) -> torch.device:
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def load_generator(repo_dir: Path, checkpoint: Path, device: torch.device):
    import torch

    sys.path.insert(0, str(repo_dir.resolve()))
    from model_vc import Generator  # type: ignore

    model = Generator(32, 256, 512, 32).eval().to(device)
    checkpoint_data = torch.load(checkpoint, map_location=device)
    state_dict = checkpoint_data["model"] if isinstance(checkpoint_data, dict) and "model" in checkpoint_data else checkpoint_data
    model.load_state_dict(clean_state_dict(state_dict))
    return model


def load_speaker_encoder(repo_dir: Path, checkpoint: Path, device: torch.device):
    import torch

    sys.path.insert(0, str(repo_dir.resolve()))
    from model_bl import D_VECTOR  # type: ignore

    model = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().to(device)
    checkpoint_data = torch.load(checkpoint, map_location=device)
    state_dict = checkpoint_data["model_b"] if isinstance(checkpoint_data, dict) and "model_b" in checkpoint_data else checkpoint_data
    model.load_state_dict(clean_state_dict(state_dict))
    return model


def speaker_embedding(
    encoder,
    mel: np.ndarray,
    device: torch.device,
    len_crop: int = 128,
    num_crops: int = 10,
) -> np.ndarray:
    import torch

    if mel.shape[0] < len_crop:
        mel = np.pad(mel, ((0, len_crop - mel.shape[0]), (0, 0)), "constant")

    if mel.shape[0] == len_crop:
        starts = [0] * num_crops
    else:
        starts = np.linspace(0, mel.shape[0] - len_crop, num=num_crops, dtype=np.int64).tolist()

    embs = []
    with torch.no_grad():
        for start in starts:
            crop = torch.from_numpy(mel[start : start + len_crop][np.newaxis, :, :]).to(device)
            embs.append(encoder(crop).detach().squeeze(0).cpu().numpy())
    emb = np.mean(embs, axis=0).astype(np.float32)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def convert_mel(generator, source_mel: np.ndarray, source_emb: np.ndarray, target_emb: np.ndarray, device: torch.device) -> np.ndarray:
    import torch

    padded, len_pad = pad_seq(source_mel)
    utterance = torch.from_numpy(padded[np.newaxis, :, :]).to(device)
    emb_org = torch.from_numpy(source_emb[np.newaxis, :]).to(device)
    emb_trg = torch.from_numpy(target_emb[np.newaxis, :]).to(device)
    with torch.no_grad():
        _, converted, _ = generator(utterance, emb_org, emb_trg)
    converted_mel = converted[0, 0, :, :].detach().cpu().numpy()
    return converted_mel if len_pad == 0 else converted_mel[:-len_pad]


def autovc_mel_to_linear_mel(mel: np.ndarray) -> np.ndarray:
    mel_db = mel * 100 - 100
    return np.power(10.0, (mel_db + 16) / 20.0)


def synthesize_griffin_lim(mel: np.ndarray, n_iter: int) -> np.ndarray:
    import librosa

    linear_mel = autovc_mel_to_linear_mel(mel).T
    return librosa.feature.inverse.mel_to_audio(
        linear_mel,
        sr=SAMPLE_RATE,
        n_fft=FFT_LENGTH,
        hop_length=HOP_LENGTH,
        fmin=MEL_FMIN,
        fmax=MEL_FMAX,
        power=1.0,
        n_iter=n_iter,
    )


def load_wavenet_vocoder(repo_dir: Path, checkpoint: Path, device: torch.device):
    import torch

    sys.path.insert(0, str(repo_dir.resolve()))
    import synthesis  # type: ignore

    synthesis.device = device
    model = synthesis.build_model().to(device)
    checkpoint_data = torch.load(checkpoint, map_location=device)
    state_dict = checkpoint_data["state_dict"] if isinstance(checkpoint_data, dict) and "state_dict" in checkpoint_data else checkpoint_data
    model.load_state_dict(state_dict)
    return synthesis, model


def write_wav(path: Path, wav: np.ndarray) -> None:
    wav = np.asarray(wav, dtype=np.float32)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak * 0.99
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav, SAMPLE_RATE)


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {description}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    baselines_root = Path(__file__).resolve().parents[1]
    repo_dir = baselines_root / "repos" / "autovc"
    parser.add_argument("--repo-dir", type=Path, default=repo_dir)
    parser.add_argument("--manifest", type=Path, default=baselines_root / "manifests" / "conversion_manifest.csv")
    parser.add_argument("--target-refs", type=Path, default=baselines_root / "manifests" / "target_references.csv")
    parser.add_argument("--autovc-checkpoint", type=Path, default=repo_dir / "autovc.ckpt")
    parser.add_argument("--speaker-encoder-checkpoint", type=Path, default=repo_dir / "3000000-BL.ckpt")
    parser.add_argument("--vocoder-checkpoint", type=Path, default=repo_dir / "checkpoint_step001000000_ema.pth")
    parser.add_argument("--dest-root", type=Path, default=baselines_root / "autovc" / "generated_wavs")
    parser.add_argument("--pairs", nargs="*", help="Optional pair filter, e.g. bdl_to_clb slt_to_rms")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--vocoder", choices=["wavenet", "griffin-lim"], default="wavenet")
    parser.add_argument("--griffin-lim-iters", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of rows for smoke tests.")
    args = parser.parse_args()

    require_path(args.manifest, "conversion manifest")
    require_path(args.target_refs, "target reference manifest")
    require_path(args.autovc_checkpoint, "AutoVC checkpoint")
    require_path(args.speaker_encoder_checkpoint, "AutoVC speaker encoder checkpoint")
    if args.vocoder == "wavenet":
        require_path(args.vocoder_checkpoint, "AutoVC WaveNet vocoder checkpoint")

    device = choose_device(args.device)
    manifest = pd.read_csv(args.manifest)
    if args.pairs:
        manifest = manifest[manifest["pair"].isin(args.pairs)]
    if args.limit is not None:
        manifest = manifest.head(args.limit)

    refs = pd.read_csv(args.target_refs).set_index("speaker")["reference_wav"].to_dict()
    basis = mel_basis()
    generator = load_generator(args.repo_dir, args.autovc_checkpoint, device)
    speaker_encoder = load_speaker_encoder(args.repo_dir, args.speaker_encoder_checkpoint, device)

    vocoder_bundle = None
    if args.vocoder == "wavenet":
        vocoder_bundle = load_wavenet_vocoder(args.repo_dir, args.vocoder_checkpoint, device)

    speaker_emb_cache: dict[str, np.ndarray] = {}

    def get_embedding(speaker: str, wav_path: Path) -> np.ndarray:
        cache_key = f"{speaker}:{wav_path}"
        if cache_key not in speaker_emb_cache:
            speaker_emb_cache[cache_key] = speaker_embedding(
                speaker_encoder,
                wav_to_autovc_mel(wav_path, basis),
                device,
            )
        return speaker_emb_cache[cache_key]

    written = 0
    for row in manifest.itertuples(index=False):
        source_wav = Path(row.src_wav)
        target_ref = Path(refs[row.tgt_spk])
        source_mel = wav_to_autovc_mel(source_wav, basis)
        source_emb = get_embedding(row.src_spk, source_wav)
        target_emb = get_embedding(row.tgt_spk, target_ref)
        converted_mel = convert_mel(generator, source_mel, source_emb, target_emb, device)

        if args.vocoder == "wavenet":
            synthesis, vocoder = vocoder_bundle
            wav = synthesis.wavegen(vocoder, c=converted_mel)
        else:
            wav = synthesize_griffin_lim(converted_mel, args.griffin_lim_iters)

        dest = args.dest_root / row.pair / f"{row.utt_id}.wav"
        write_wav(dest, wav)
        written += 1
        print(f"{written}: {row.pair}/{row.utt_id}.wav")

    print(f"Wrote {written} AutoVC outputs to {args.dest_root}")


if __name__ == "__main__":
    main()
