"""Audio IO - waveform tensor processing only (file decoding handled externally by load_audio)."""

import random
from io import IOBase
from pathlib import Path
from typing import Mapping

import torch.nn.functional as F
import torchaudio
from pyannote.core import Segment
from torch import Tensor

AudioFile = str | Path | IOBase | Mapping


class Audio:
    """Audio IO: downmix, resample, crop on in-memory waveform tensors."""

    PRECISION = 0.001

    @staticmethod
    def power_normalize(waveform: Tensor) -> Tensor:
        rms = waveform.square().mean(dim=-1, keepdim=True).sqrt()
        return waveform / (rms + 1e-8)

    @staticmethod
    def validate_file(file: AudioFile) -> Mapping:
        if isinstance(file, Mapping):
            pass
        elif isinstance(file, (str, Path)):
            file = {"audio": str(file), "uri": Path(file).stem}
        elif isinstance(file, IOBase):
            return {"audio": file, "uri": "stream"}
        else:
            raise ValueError("Unsupported audio file type.")

        if "waveform" in file:
            waveform: Tensor = file["waveform"]
            if len(waveform.shape) != 2 or waveform.shape[0] > waveform.shape[1]:
                raise ValueError("'waveform' must be (channel, time) torch Tensor.")
            if file.get("sample_rate") is None:
                raise ValueError("'waveform' must be provided with 'sample_rate'.")
            file.setdefault("uri", "waveform")

        elif "audio" in file:
            if isinstance(file["audio"], IOBase):
                return file
            path = Path(file["audio"])
            if not path.is_file():
                raise ValueError(f"File {path} does not exist")
            file.setdefault("uri", path.stem)
        else:
            raise ValueError("Neither 'waveform' nor 'audio' is available.")

        return file

    def __init__(self, sample_rate: int | None = None, mono=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.mono = mono

    def downmix_and_resample(
        self, waveform: Tensor, sample_rate: int, channel: int | None = None
    ) -> tuple[Tensor, int]:
        if channel is not None:
            waveform = waveform[channel : channel + 1]

        num_channels = waveform.shape[0]
        if num_channels > 1:
            if self.mono == "random":
                channel = random.randint(0, num_channels - 1)
                waveform = waveform[channel : channel + 1]
            elif self.mono == "downmix":
                waveform = waveform.mean(dim=0, keepdim=True)

        if (self.sample_rate is not None) and (self.sample_rate != sample_rate):
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.sample_rate
            )
            sample_rate = self.sample_rate

        return waveform, sample_rate

    def get_duration(self, file: AudioFile) -> float:
        file = self.validate_file(file)
        if "waveform" in file:
            return file["waveform"].shape[1] / file["sample_rate"]
        raise ValueError("'waveform' key required to get duration.")

    def get_num_samples(self, duration: float, sample_rate: int | None = None) -> int:
        sample_rate = sample_rate or self.sample_rate
        if sample_rate is None:
            raise ValueError("`sample_rate` must be provided.")
        return round(duration * sample_rate)

    def __call__(self, file: AudioFile) -> tuple[Tensor, int]:
        file = self.validate_file(file)
        channel = file.get("channel", None)

        if "waveform" in file:
            return self.downmix_and_resample(file["waveform"], file["sample_rate"], channel=channel)

        raise ValueError("'waveform' key required. Use load_audio() to decode files first.")

    def crop(
        self,
        file: AudioFile,
        segment: Segment,
        mode="raise",
    ) -> tuple[Tensor, int]:
        file = self.validate_file(file)
        channel = file.get("channel", None)

        if "waveform" not in file:
            raise ValueError("'waveform' key required. Use load_audio() to decode files first.")

        waveform = file["waveform"]
        _, num_samples = waveform.shape
        sample_rate = file["sample_rate"]
        duration = num_samples / sample_rate

        start_sample = self.get_num_samples(segment.start, sample_rate)
        pad_start = max(0, -start_sample)
        if start_sample < 0:
            if mode == "raise":
                raise ValueError(f"Negative start time (t={segment.start:.3f}s)")
            start_sample = 0

        end_sample = self.get_num_samples(segment.end, sample_rate)
        pad_end = max(end_sample, num_samples) - num_samples
        if end_sample >= num_samples:
            if mode == "raise":
                raise ValueError(f"End time (t={segment.end:.3f}s) > duration ({duration:.3f}s).")
            end_sample = num_samples

        data = waveform[:, start_sample:end_sample]
        data = F.pad(data, (pad_start, pad_end))
        return self.downmix_and_resample(data, sample_rate, channel=channel)
