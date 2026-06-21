#!/usr/bin/env python3
"""Compare VoiceGrad outputs against VC baseline model outputs.

The script supports two workflows:
1. Reuse existing metric CSVs in each model result directory.
2. Compute MCD/LFC from matched generated WAVs and target-speaker reference WAVs.

Folder convention for generated WAVs:
    <model_result_dir>/generated_wavs/<src>_to_<tgt>/<utt_id>.wav

Existing CSV convention, matching the current VoiceGrad notebook outputs:
    mcd_lfc_detail.csv
    cer_detail.csv
    pmos_dnsmos_detail.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from scipy.spatial.distance import cdist


os.environ.setdefault("LIBROSA_CACHE_LEVEL", "0")

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_VOICEGRAD_ROOT = PIPELINE_DIR.parent

DEFAULT_MODELS = {
    "VoiceGrad-DPM-BNF": "checkpoints/ckpt1000_closedset_test_result",
}

METRIC_DIRECTIONS = {
    "mcd_db": "lower",
    "lfc": "higher",
    "cer": "lower",
    "pmos": "higher",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    result_dir: Path


def parse_model_specs(values: list[str], base_dir: Path) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    if not values:
        values = [f"{name}={path}" for name, path in DEFAULT_MODELS.items()]

    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --model value {value!r}; expected NAME=DIR")
        name, raw_dir = value.split("=", 1)
        name = name.strip()
        result_dir = Path(raw_dir).expanduser()
        if not result_dir.is_absolute():
            result_dir = base_dir / result_dir
        if not name:
            raise SystemExit(f"Invalid --model value {value!r}; model name is empty")
        specs.append(ModelSpec(name=name, result_dir=result_dir))
    return specs


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("src_spk", "tgt_spk", "utt_id"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def load_existing_details(model: ModelSpec) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    mcd_lfc = read_csv_if_exists(model.result_dir / "mcd_lfc_detail.csv")
    if mcd_lfc is not None:
        keep = [c for c in ("src_spk", "tgt_spk", "utt_id", "gen_wav_path", "ref_wav_path", "mcd_db", "lfc") if c in mcd_lfc.columns]
        parts.append(normalize_key_columns(mcd_lfc[keep]))

    cer = read_csv_if_exists(model.result_dir / "cer_detail.csv")
    if cer is not None:
        keep = [c for c in ("src_spk", "tgt_spk", "utt_id", "cer", "ref", "hyp", "wav_path") if c in cer.columns]
        part = normalize_key_columns(cer[keep])
        if "wav_path" in part.columns:
            part = part.rename(columns={"wav_path": "cer_wav_path"})
        parts.append(part)

    pmos = read_csv_if_exists(model.result_dir / "pmos_dnsmos_detail.csv")
    if pmos is not None:
        keep = [c for c in ("src_spk", "tgt_spk", "utt_id", "pmos", "mos", "wav_path") if c in pmos.columns]
        part = normalize_key_columns(pmos[keep])
        if "wav_path" in part.columns:
            part = part.rename(columns={"wav_path": "pmos_wav_path"})
        parts.append(part)

    if not parts:
        return pd.DataFrame()

    detail = parts[0]
    for part in parts[1:]:
        detail = detail.merge(part, on=["src_spk", "tgt_spk", "utt_id"], how="outer")
    detail.insert(0, "model", model.name)
    return detail


def iter_generated_wavs(result_dir: Path) -> Iterable[dict[str, str]]:
    root = result_dir / "generated_wavs"
    if not root.exists():
        return
    pattern = re.compile(r"^(?P<src>[^/]+)_to_(?P<tgt>[^/]+)$")
    for wav_path in sorted(root.glob("*/*.wav")):
        match = pattern.match(wav_path.parent.name)
        if not match:
            continue
        yield {
            "src_spk": match.group("src"),
            "tgt_spk": match.group("tgt"),
            "utt_id": wav_path.stem,
            "gen_wav_path": str(wav_path),
        }


def reference_wav_path(ref_wav_root: Path, tgt_spk: str, utt_id: str) -> Path:
    return ref_wav_root / tgt_spk / "wav" / f"{utt_id}.wav"


def require_soundfile():
    try:
        import soundfile as sf  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Audio metric computation needs soundfile. Install optional dependencies with:\n"
            "  python3 -m pip install -r voicegrad/evaluation_pipeline/eval_requirements.txt"
        ) from exc
    return sf


def load_audio(path: Path, sr: int):
    sf = require_soundfile()
    y, file_sr = sf.read(path, always_2d=False)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if file_sr != sr:
        gcd = math.gcd(int(file_sr), int(sr))
        y = resample_poly(y, sr // gcd, file_sr // gcd).astype(np.float64)
    if y.size == 0:
        raise ValueError(f"Empty audio: {path}")
    return y


def require_world_dependencies():
    try:
        import pyworld as pw  # type: ignore
        import pysptk  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Paper-style MCD/LFC computation needs pyworld and pysptk. "
            "Use the baseline environment or install them, e.g.:\n"
            "  conda activate voicegrad-baselines"
        ) from exc
    return pw, pysptk


def dtw_path_from_cost(cost: np.ndarray) -> np.ndarray:
    try:
        from numba import njit  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Paper-style MCD/LFC computation needs numba for fast DTW. "
            "Install it or use the voicegrad-baselines environment."
        ) from exc

    @njit(cache=False)
    def _dtw(cost_matrix):
        n, m = cost_matrix.shape
        acc = np.empty((n + 1, m + 1), dtype=np.float64)
        back = np.zeros((n, m), dtype=np.int8)
        for i in range(n + 1):
            for j in range(m + 1):
                acc[i, j] = np.inf
        acc[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                diag = acc[i - 1, j - 1]
                up = acc[i - 1, j]
                left = acc[i, j - 1]
                if diag <= up and diag <= left:
                    best = diag
                    step = 0
                elif up <= left:
                    best = up
                    step = 1
                else:
                    best = left
                    step = 2
                acc[i, j] = cost_matrix[i - 1, j - 1] + best
                back[i - 1, j - 1] = step

        max_len = n + m
        path = np.empty((max_len, 2), dtype=np.int64)
        path_len = 0
        i = n - 1
        j = m - 1
        while i >= 0 and j >= 0:
            path[path_len, 0] = i
            path[path_len, 1] = j
            path_len += 1
            step = back[i, j]
            if step == 0:
                i -= 1
                j -= 1
            elif step == 1:
                i -= 1
            else:
                j -= 1

        out = np.empty((path_len, 2), dtype=np.int64)
        for k in range(path_len):
            out[k, 0] = path[path_len - 1 - k, 0]
            out[k, 1] = path[path_len - 1 - k, 1]
        return out

    if cost.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return _dtw(cost.astype(np.float64))


def extract_f0_mcep(
    wav_path: Path,
    sr: int,
    frame_period_ms: float,
    mcep_order: int,
    mcep_alpha: float,
) -> dict[str, np.ndarray]:
    pw, pysptk = require_world_dependencies()
    wav = load_audio(wav_path, sr)
    f0, timeaxis = pw.harvest(
        wav,
        sr,
        frame_period=frame_period_ms,
        f0_floor=20.0,
        f0_ceil=600.0,
    )
    sp = pw.cheaptrick(wav, f0, timeaxis, sr)
    mcep = pysptk.sp2mc(sp, order=mcep_order, alpha=mcep_alpha)
    return {"f0": f0.astype(np.float64), "mcep": mcep.astype(np.float64)}


def compute_mcd_lfc(
    gen_wav: Path,
    ref_wav: Path,
    sr: int,
    frame_period_ms: float,
    mcep_order: int,
    mcep_alpha: float,
) -> tuple[float, float]:
    gen = extract_f0_mcep(gen_wav, sr, frame_period_ms, mcep_order, mcep_alpha)
    ref = extract_f0_mcep(ref_wav, sr, frame_period_ms, mcep_order, mcep_alpha)

    cost = cdist(gen["mcep"][:, 1:], ref["mcep"][:, 1:], metric="euclidean")
    path = dtw_path_from_cost(cost)
    if len(path) == 0:
        return float("nan"), float("nan")

    mcd_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
    dists = []
    xs = []
    ys = []
    for i, j in path:
        diff = gen["mcep"][i, 1:] - ref["mcep"][j, 1:]
        dists.append(mcd_const * np.sqrt(np.sum(diff * diff)))
        if gen["f0"][i] > 0 and ref["f0"][j] > 0:
            xs.append(np.log(gen["f0"][i]))
            ys.append(np.log(ref["f0"][j]))

    mcd = float(np.mean(dists)) if dists else float("nan")
    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys)
    if len(xs_arr) < 2 or np.std(xs_arr) < 1e-12 or np.std(ys_arr) < 1e-12:
        lfc = float("nan")
    else:
        lfc = float(np.corrcoef(xs_arr, ys_arr)[0, 1])
    return mcd, lfc


def compute_audio_details(
    model: ModelSpec,
    ref_wav_root: Path,
    sr: int,
    frame_period_ms: float,
    mcep_order: int,
    mcep_alpha: float,
) -> pd.DataFrame:
    rows = []
    for row in iter_generated_wavs(model.result_dir):
        ref_path = reference_wav_path(ref_wav_root, row["tgt_spk"], row["utt_id"])
        gen_path = Path(row["gen_wav_path"])
        output = {
            "model": model.name,
            **row,
            "ref_wav_path": str(ref_path),
            "mcd_db": np.nan,
            "lfc": np.nan,
            "error": "",
        }
        if not ref_path.exists():
            output["error"] = f"missing reference wav: {ref_path}"
        else:
            try:
                output["mcd_db"], output["lfc"] = compute_mcd_lfc(
                    gen_path,
                    ref_path,
                    sr=sr,
                    frame_period_ms=frame_period_ms,
                    mcep_order=mcep_order,
                    mcep_alpha=mcep_alpha,
                )
            except Exception as exc:  # Keep one bad file from stopping the comparison.
                output["error"] = str(exc)
        rows.append(output)
    return pd.DataFrame(rows)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_ref_map(text_path: Path) -> dict[str, str]:
    if not text_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {text_path}")
    ref_map: dict[str, str] = {}
    for raw_line in text_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r'\(\s*(arctic_[ab]\d+)\s+"(.+)"\s*\)', line)
        if match:
            ref_map[match.group(1)] = normalize_text(match.group(2))
            continue
        if "|" in line:
            utt_id, text = line.split("|", 1)
            ref_map[utt_id.strip()] = normalize_text(text)
            continue
        if "\t" in line:
            utt_id, text = line.split("\t", 1)
            ref_map[utt_id.strip()] = normalize_text(text)
    if not ref_map:
        raise RuntimeError(f"Failed to parse transcript file: {text_path}")
    return ref_map


def require_asr_dependencies():
    try:
        import torch  # type: ignore
        from jiwer import cer  # type: ignore
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "CER computation needs torch, transformers, and jiwer. "
            "Install them in the active environment first."
        ) from exc
    return torch, cer, Wav2Vec2ForCTC, Wav2Vec2Processor


def compute_cer_details(
    model: ModelSpec,
    text_path: Path,
    asr_model_name: str,
    sr: int,
    device: str,
) -> pd.DataFrame:
    torch, cer, Wav2Vec2ForCTC, Wav2Vec2Processor = require_asr_dependencies()
    ref_map = load_ref_map(text_path)
    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
    asr_model = Wav2Vec2ForCTC.from_pretrained(asr_model_name).to(resolved_device)
    asr_model.eval()

    rows = []
    for row in iter_generated_wavs(model.result_dir):
        wav_path = Path(row["gen_wav_path"])
        ref = ref_map.get(row["utt_id"])
        output = {
            "model": model.name,
            "src_spk": row["src_spk"],
            "tgt_spk": row["tgt_spk"],
            "utt_id": row["utt_id"],
            "ref": ref,
            "hyp": "",
            "cer": np.nan,
            "cer_wav_path": str(wav_path),
            "cer_error": "",
        }
        if ref is None:
            output["cer_error"] = f"missing transcript for {row['utt_id']}"
            rows.append(output)
            continue
        try:
            wav = load_audio(wav_path, sr)
            inputs = processor(wav, sampling_rate=sr, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(resolved_device)
            attention_mask = inputs.attention_mask.to(resolved_device) if "attention_mask" in inputs else None
            with torch.no_grad():
                logits = asr_model(input_values, attention_mask=attention_mask).logits
            pred_ids = torch.argmax(logits, dim=-1)
            hyp = normalize_text(processor.batch_decode(pred_ids)[0])
            output["hyp"] = hyp
            output["cer"] = float(cer(ref, hyp))
        except Exception as exc:
            output["cer_error"] = str(exc)
        rows.append(output)
    return pd.DataFrame(rows)


def require_pmos_dependencies():
    try:
        from speechmos import dnsmos  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "pMOS computation needs speechmos and onnxruntime. "
            "Install them in the active environment first."
        ) from exc
    return dnsmos


def compute_pmos_details(model: ModelSpec, sr: int) -> pd.DataFrame:
    dnsmos = require_pmos_dependencies()
    rows = []
    for row in iter_generated_wavs(model.result_dir):
        wav_path = Path(row["gen_wav_path"])
        output = {
            "model": model.name,
            "src_spk": row["src_spk"],
            "tgt_spk": row["tgt_spk"],
            "utt_id": row["utt_id"],
            "pmos": np.nan,
            "pmos_wav_path": str(wav_path),
            "pmos_error": "",
        }
        try:
            audio = load_audio(wav_path, sr).astype(np.float32)
            scores = dnsmos.run(audio, sr)
            output["pmos"] = float(scores["ovrl_mos"])
        except Exception as exc:
            output["pmos_error"] = str(exc)
        rows.append(output)
    return pd.DataFrame(rows)


def ci95(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return float("nan")
    return float(1.96 * values.std(ddof=1) / math.sqrt(len(values)))


def summarize(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [m for m in ("mcd_db", "lfc", "cer", "pmos", "mos") if m in detail.columns]
    if detail.empty or not metrics:
        return pd.DataFrame(), pd.DataFrame()

    group_cols = ["model"]
    pair_group_cols = ["model", "src_spk", "tgt_spk"]
    overall_rows = []
    pair_rows = []

    for keys, group in detail.groupby(group_cols, dropna=False):
        row = {"model": keys if isinstance(keys, str) else keys[0], "count": len(group)}
        for metric in metrics:
            numeric = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(numeric.mean()) if numeric.notna().any() else np.nan
            row[f"{metric}_ci95"] = ci95(numeric)
            row[f"{metric}_n"] = int(numeric.notna().sum())
        overall_rows.append(row)

    for keys, group in detail.groupby(pair_group_cols, dropna=False):
        model, src_spk, tgt_spk = keys
        row = {"model": model, "src_spk": src_spk, "tgt_spk": tgt_spk, "count": len(group)}
        for metric in metrics:
            numeric = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(numeric.mean()) if numeric.notna().any() else np.nan
            row[f"{metric}_ci95"] = ci95(numeric)
            row[f"{metric}_n"] = int(numeric.notna().sum())
        pair_rows.append(row)

    overall = pd.DataFrame(overall_rows)
    pairs = pd.DataFrame(pair_rows)
    return overall, pairs


def add_rank_columns(overall: pd.DataFrame) -> pd.DataFrame:
    out = overall.copy()
    for metric, direction in METRIC_DIRECTIONS.items():
        col = f"{metric}_mean"
        if col not in out.columns:
            continue
        ascending = direction == "lower"
        out[f"{metric}_rank"] = out[col].rank(method="min", ascending=ascending, na_option="bottom").astype("Int64")
    return out


def merge_metric_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return pd.DataFrame()
    keys = ["model", "src_spk", "tgt_spk", "utt_id"]
    merged = usable[0]
    for frame in usable[1:]:
        overlap = [c for c in frame.columns if c in merged.columns and c not in keys]
        frame_to_merge = frame.rename(columns={c: f"{c}__dup" for c in overlap})
        merged = merged.merge(frame_to_merge, on=keys, how="outer")
        for col in overlap:
            dup = f"{col}__dup"
            merged[col] = merged[col].combine_first(merged[dup])
            merged = merged.drop(columns=[dup])
    return merged


def merge_existing_and_computed(existing: pd.DataFrame, computed: pd.DataFrame, prefer_computed: bool) -> pd.DataFrame:
    if existing.empty:
        return computed
    if computed.empty:
        return existing

    keys = ["model", "src_spk", "tgt_spk", "utt_id"]
    existing_prefixed = existing.rename(columns={c: f"existing_{c}" for c in existing.columns if c not in keys})
    computed_prefixed = computed.rename(columns={c: f"computed_{c}" for c in computed.columns if c not in keys})
    merged = computed_prefixed.merge(existing_prefixed, on=keys, how="outer")

    all_base_cols = sorted(
        {
            c.removeprefix("existing_")
            for c in merged.columns
            if c.startswith("existing_")
        }
        | {
            c.removeprefix("computed_")
            for c in merged.columns
            if c.startswith("computed_")
        }
    )
    for col in all_base_cols:
        existing_col = f"existing_{col}"
        computed_col = f"computed_{col}"
        if existing_col in merged.columns and computed_col in merged.columns:
            if prefer_computed:
                merged[col] = merged[computed_col].combine_first(merged[existing_col])
            else:
                merged[col] = merged[existing_col].combine_first(merged[computed_col])
        elif existing_col in merged.columns:
            merged[col] = merged[existing_col]
        elif computed_col in merged.columns:
            merged[col] = merged[computed_col]

    drop_cols = [c for c in merged.columns if c.startswith("existing_") or c.startswith("computed_")]
    return merged.drop(columns=drop_cols)


def write_metric_manifest(path: Path) -> None:
    rows = [
        ("MCD", "mcd_db", "Mel cepstral distortion between converted and target speech", "lower"),
        ("LFC", "lfc", "Correlation of log-F0 contour between converted and target speech", "higher"),
        ("CER", "cer", "Character error rate from ASR transcript versus reference text", "lower"),
        ("pMOS/MOS", "pmos or mos", "Predicted or human mean opinion score for naturalness/audio quality", "higher"),
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "column", "meaning", "better"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model result directory as NAME=DIR. Repeat for StarGAN-VC, AutoVC, PPG-VC, Diff-VC, VoiceGrad.",
    )
    parser.add_argument("--voicegrad-root", type=Path, default=DEFAULT_VOICEGRAD_ROOT)
    parser.add_argument("--ref-wav-root", type=Path, default=None, help="Reference wav root, default: <voicegrad-root>/data/wav")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory, default: voicegrad/evaluation_pipeline/comparison_eval",
    )
    parser.add_argument(
        "--compute-audio",
        action="store_true",
        help="Compute paper-style MCD/LFC from generated_wavs folders.",
    )
    parser.add_argument(
        "--compute-cer",
        action="store_true",
        help="Compute CER with wav2vec2 from generated_wavs folders.",
    )
    parser.add_argument(
        "--compute-pmos",
        action="store_true",
        help="Compute pMOS with speechmos DNSMOS from generated_wavs folders.",
    )
    parser.add_argument(
        "--prefer-computed",
        action="store_true",
        help="When existing CSV metrics and computed audio metrics both exist, use computed MCD/LFC.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-period-ms", type=float, default=10.0)
    parser.add_argument("--mcep-order", type=int, default=24)
    parser.add_argument("--mcep-alpha", type=float, default=0.42)
    parser.add_argument("--text-path", type=Path, default=None, help="Transcript file, default: <voicegrad-root>/data/cmuarctic.data.txt")
    parser.add_argument("--asr-model", default="facebook/wav2vec2-large-960h-lv60-self")
    parser.add_argument("--device", default="auto", help="Device for ASR, e.g. auto, cpu, cuda")
    parser.add_argument(
        "--write-model-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write computed mcd_lfc_detail.csv, cer_detail.csv, and pmos_dnsmos_detail.csv into each model result directory.",
    )
    args = parser.parse_args()

    voicegrad_root = args.voicegrad_root.resolve()
    ref_wav_root = (args.ref_wav_root or voicegrad_root / "data" / "wav").resolve()
    text_path = (args.text_path or voicegrad_root / "data" / "cmuarctic.data.txt").resolve()
    out_dir = (args.out_dir or PIPELINE_DIR / "comparison_eval").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    models = parse_model_specs(args.model, voicegrad_root)
    detail_frames: list[pd.DataFrame] = []
    for model in models:
        existing = load_existing_details(model)
        computed_parts: list[pd.DataFrame] = []
        if not existing.empty:
            print(f"Loaded existing metrics for {model.name}: {len(existing)} rows")
        if args.compute_audio:
            mcd_lfc = compute_audio_details(
                model,
                ref_wav_root=ref_wav_root,
                sr=args.sample_rate,
                frame_period_ms=args.frame_period_ms,
                mcep_order=args.mcep_order,
                mcep_alpha=args.mcep_alpha,
            )
            if not mcd_lfc.empty:
                computed_parts.append(mcd_lfc)
                if args.write_model_metrics:
                    cols = [c for c in ("src_spk", "tgt_spk", "utt_id", "gen_wav_path", "ref_wav_path", "mcd_db", "lfc", "error") if c in mcd_lfc.columns]
                    mcd_lfc[cols].to_csv(model.result_dir / "mcd_lfc_detail.csv", index=False, encoding="utf-8-sig")
                print(f"Computed MCD/LFC for {model.name}: {len(mcd_lfc)} rows")
        if args.compute_cer:
            cer_df = compute_cer_details(
                model,
                text_path=text_path,
                asr_model_name=args.asr_model,
                sr=args.sample_rate,
                device=args.device,
            )
            if not cer_df.empty:
                computed_parts.append(cer_df)
                if args.write_model_metrics:
                    cols = [c for c in ("src_spk", "tgt_spk", "utt_id", "ref", "hyp", "cer", "cer_wav_path", "cer_error") if c in cer_df.columns]
                    out = cer_df[cols].rename(columns={"cer_wav_path": "wav_path"})
                    out.to_csv(model.result_dir / "cer_detail.csv", index=False, encoding="utf-8-sig")
                print(f"Computed CER for {model.name}: {len(cer_df)} rows")
        if args.compute_pmos:
            pmos_df = compute_pmos_details(model, sr=args.sample_rate)
            if not pmos_df.empty:
                computed_parts.append(pmos_df)
                if args.write_model_metrics:
                    cols = [c for c in ("src_spk", "tgt_spk", "utt_id", "pmos", "pmos_wav_path", "pmos_error") if c in pmos_df.columns]
                    out = pmos_df[cols].rename(columns={"pmos_wav_path": "wav_path"})
                    out.to_csv(model.result_dir / "pmos_dnsmos_detail.csv", index=False, encoding="utf-8-sig")
                print(f"Computed pMOS for {model.name}: {len(pmos_df)} rows")
        computed = merge_metric_frames(computed_parts)
        merged = merge_existing_and_computed(existing, computed, prefer_computed=args.prefer_computed)
        if not merged.empty:
            detail_frames.append(merged)

    if not detail_frames:
        raise SystemExit(
            "No evaluation rows found. Add result CSVs or generated_wavs folders, then pass --model NAME=DIR."
        )

    detail = pd.concat(detail_frames, ignore_index=True, sort=False)
    key_cols = ["model", "src_spk", "tgt_spk", "utt_id"]
    sort_cols = [c for c in key_cols if c in detail.columns]
    if sort_cols:
        detail = detail.sort_values(sort_cols)

    overall, pairs = summarize(detail)
    overall = add_rank_columns(overall)

    detail.to_csv(out_dir / "model_metric_detail.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(out_dir / "model_metric_summary.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(out_dir / "model_metric_summary_by_pair.csv", index=False, encoding="utf-8-sig")
    write_metric_manifest(out_dir / "metric_manifest.csv")

    print(f"Wrote detail: {out_dir / 'model_metric_detail.csv'}")
    print(f"Wrote summary: {out_dir / 'model_metric_summary.csv'}")
    print(f"Wrote pair summary: {out_dir / 'model_metric_summary_by_pair.csv'}")
    print(f"Wrote manifest: {out_dir / 'metric_manifest.csv'}")


if __name__ == "__main__":
    main()
