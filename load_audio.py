import subprocess
from pathlib import Path

import torch


def load_audio(audio_path: str | Path, *, use_filter: bool = False) -> torch.Tensor:
    AUDIO_FILTER = (
        "-af "
        "aresample=resampler=soxr:precision=28,"
        "highpass=f=60,"
        "dynaudnorm=f=150:g=11,"
        "acompressor=threshold=0.2:ratio=2:attack=200:release=1000:makeup=1 "
    )
    ffmpeg_cmd = " ".join(
        [
            "ffmpeg -nostdin -loglevel warning -threads 0 -y",
            AUDIO_FILTER if use_filter else "",
            "-f f32le -ac 1 -acodec pcm_f32le -ar 16000 -",
        ]
    ).split()
    ffmpeg_cmd.extend(["-i", str(audio_path)])

    raw_bytes = subprocess.run(
        ffmpeg_cmd, capture_output=True, check=True, timeout=60 * 5
    ).stdout

    return torch.frombuffer(bytearray(raw_bytes), dtype=torch.float32).unsqueeze(0)  # (1, num_samples)
