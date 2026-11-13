from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

SAMPLE_RATE = 16000
MIN_TAIL = 1.0
MIN_MAX_CHUNK = 10.0


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Chunk:
    path: Path
    offset: float
    duration: float


def require_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found on PATH; please install ffmpeg")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr.strip()[-500:]}")
    return p


def probe_duration(path: Path) -> float:
    out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1", str(path)]).stdout.strip()
    try:
        return float(out)
    except ValueError:
        raise ValueError(f"Could not read duration of {path.name}")


def normalize(src: Path, workdir: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")
    if not src.is_file():
        raise ValueError(f"Not a file: {src}")
    if src.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {src.name}")
    dst = workdir / "normalized.wav"
    _run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src), "-vn",
          "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)])
    if not dst.exists() or probe_duration(dst) < 0.1:
        raise ValueError("Audio is empty or unreadable")
    return dst


def detect_silences(wav: Path, noise_db: int = -35, min_silence: float = 0.4) -> list[float]:
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "info", "-i", str(wav), "-af",
         f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
        capture_output=True, text=True, errors="replace")
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", p.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: (-?[\d.]+)", p.stderr)]
    return [(s + e) / 2 for s, e in zip(starts, ends)]


def plan_cuts(total: float, silences: list[float], target: float, min_len: float, max_len: float) -> list[float]:
    cuts = [0.0]
    cursor = 0.0
    while total - cursor > max_len:
        candidates = [s for s in silences if cursor + min_len <= s <= cursor + max_len]
        if candidates:
            nxt = min(candidates, key=lambda s: abs(s - (cursor + target)))
        else:
            nxt = cursor + max_len
        cuts.append(nxt)
        cursor = nxt
    if len(cuts) > 1 and total - cuts[-1] < MIN_TAIL:
        cuts.pop()
    cuts.append(total)
    return cuts


def make_chunks(wav: Path, workdir: Path, target=300.0, min_len=60.0, max_len=600.0) -> list[Chunk]:
    if max_len < MIN_MAX_CHUNK:
        raise ValueError(f"max chunk length must be at least {MIN_MAX_CHUNK:g} seconds")
    total = probe_duration(wav)
    if total <= max_len:
        return [Chunk(wav, 0.0, total)]
    min_len = min(min_len, max_len / 2)
    target = min(target, max_len / 2)
    cuts = plan_cuts(total, detect_silences(wav), target, min_len, max_len)
    chunks = []
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        out = workdir / f"chunk_{i:04d}.wav"
        _run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}",
              "-i", str(wav), "-c", "copy", str(out)])
        chunks.append(Chunk(out, a, b - a))
    return chunks


def load_wav(path: Path):
    import numpy as np
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("Unexpected WAV format after normalization")
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


class Backend(Protocol):
    parallel_safe: bool

    def transcribe(self, wav: Path) -> list[Segment]:
        ...


class WhisperBackend:
    parallel_safe = False

    def __init__(self, model="small", device="auto", compute_type="int8", language=None):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.language = language

    def transcribe(self, wav: Path) -> list[Segment]:
        audio = load_wav(wav)
        segs, _info = self.model.transcribe(
            audio, language=self.language, vad_filter=True,
            condition_on_previous_text=False,
        )
        return [Segment(s.start, s.end, s.text) for s in segs]


class MockBackend:
    parallel_safe = True

    def transcribe(self, wav: Path) -> list[Segment]:
        dur, step, out, t, i = probe_duration(wav), 5.0, [], 0.0, 0
        while t < dur:
            e = min(t + step, dur)
            out.append(Segment(t, e, f" This is mock sentence {i}."))
            t, i = e, i + 1
        return out


def get_backend(name: str, **kw) -> Backend:
    if name == "whisper":
        return WhisperBackend(**kw)
    if name == "mock":
        return MockBackend()
    raise ValueError(f"unknown backend {name!r}")


def postprocess(segs: list[Segment], merge_gap=0.3, max_merged=15.0, repeat_gap=1.0) -> list[Segment]:
    cleaned: list[Segment] = []
    for s in sorted(segs, key=lambda x: (x.start, x.end)):
        text = re.sub(r"\s+", " ", s.text).strip()
        if not text:
            continue
        prev = cleaned[-1] if cleaned else None
        if prev and text.lower() == prev.text.lower() and s.start - prev.end <= repeat_gap:
            prev.end = max(prev.end, s.end)
            continue
        if (prev and s.start - prev.end <= merge_gap
                and not prev.text.endswith((".", "?", "!"))
                and s.end - prev.start <= max_merged):
            prev.text += " " + text
            prev.end = max(prev.end, s.end)
            continue
        cleaned.append(Segment(s.start, s.end, text))
    return cleaned


def transcribe_file(path: str | Path, backend: Backend, workers: int = 1, **chunk_kw) -> dict:
    require_tools()
    path = Path(path)
    if not getattr(backend, "parallel_safe", False):
        workers = 1
    workers = max(1, workers)
    tmp = Path(tempfile.mkdtemp(prefix="asr_"))
    try:
        wav = normalize(path, tmp)
        chunks = make_chunks(wav, tmp, **chunk_kw)

        def run(c: Chunk) -> list[Segment]:
            return [Segment(s.start + c.offset, s.end + c.offset, s.text) for s in backend.transcribe(c.path)]

        with ThreadPoolExecutor(max_workers=workers) as ex:
            per_chunk = list(ex.map(run, chunks))

        segs = postprocess([s for group in per_chunk for s in group])
        return {
            "source": path.name,
            "duration": round(probe_duration(wav), 3),
            "num_chunks": len(chunks),
            "segments": [{"id": i, "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
                         for i, s in enumerate(segs)],
            "text": " ".join(s.text for s in segs),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def to_srt(result: dict) -> str:
    return "\n".join(f"{s['id'] + 1}\n{_ts(s['start'])} --> {_ts(s['end'])}\n{s['text']}\n"
                     for s in result["segments"])


def write_text(path: str, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def main(argv=None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="Audio file -> transcript with per-segment timestamps")
    ap.add_argument("audio")
    ap.add_argument("--backend", default="whisper", choices=["whisper", "mock"])
    ap.add_argument("--model", default="small", help="whisper model size (tiny/base/small/medium/large-v3)")
    ap.add_argument("--language", default=None, help="e.g. en, ur, pa; default auto-detect")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-chunk", type=float, default=600.0, help="hard max chunk seconds (>= 10)")
    ap.add_argument("-o", "--output", help="write JSON here (default: stdout)")
    ap.add_argument("--srt", help="also write SRT subtitles here")
    a = ap.parse_args(argv)

    language = None if a.language in (None, "", "auto") else a.language
    kw = {"model": a.model, "language": language} if a.backend == "whisper" else {}
    result = transcribe_file(a.audio, get_backend(a.backend, **kw), workers=a.workers, max_len=a.max_chunk)

    js = json.dumps(result, indent=2, ensure_ascii=False)
    if a.output:
        write_text(a.output, js)
    else:
        print(js)
    if a.srt:
        write_text(a.srt, to_srt(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"error: {e}")