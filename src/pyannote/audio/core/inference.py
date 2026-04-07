"""Inference - minimal version without lightning dependency"""

import warnings
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import Model, Resolution, Specifications
from pyannote.audio.utils.powerset import Powerset
from pyannote.audio.utils.reproducibility import fix_reproducibility
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature


def _map_with_specifications(specifications, func, *iterables):
    if isinstance(specifications, Specifications):
        return func(*iterables, specifications=specifications)
    return tuple(
        func(*i, specifications=s) for s, *i in zip(specifications, *iterables)
    )


class BaseInference:
    pass


class Inference(BaseInference):
    """Inference with sliding window aggregation."""

    def __init__(
        self,
        model: Model,
        window: str = "sliding",
        duration: float | None = None,
        step: float | None = None,
        pre_aggregation_hook: Callable[[np.ndarray], np.ndarray] = None,
        skip_aggregation: bool = False,
        skip_conversion: bool = False,
        device: torch.device | None = None,
        batch_size: int = 32,
    ):
        self.model = model

        if device is None:
            device = next(model.parameters()).device
        self.device = device

        self.model.eval()
        self.model.to(self.device)

        specifications = self.model.specifications

        if window not in ["sliding", "whole"]:
            raise ValueError('`window` must be "sliding" or "whole".')

        self.window = window

        training_duration = next(iter(specifications)).duration
        duration = duration or training_duration
        if training_duration != duration:
            warnings.warn(
                f"Model was trained with {training_duration:g}s chunks, and you requested "
                f"{duration:g}s chunks for inference."
            )
        self.duration = duration

        self.skip_conversion = skip_conversion

        conversion = list()
        for s in specifications:
            if s.powerset and not skip_conversion:
                c = Powerset(len(s.classes), s.powerset_max_classes)
            else:
                c = nn.Identity()
            conversion.append(c.to(self.device))

        if isinstance(specifications, Specifications):
            self.conversion = conversion[0]
        else:
            self.conversion = nn.ModuleList(conversion)

        self.skip_aggregation = skip_aggregation
        self.pre_aggregation_hook = pre_aggregation_hook

        self.warm_up = next(iter(specifications)).warm_up

        step = step or (
            0.1 * self.duration if self.warm_up[0] == 0.0 else self.warm_up[0]
        )

        if step > self.duration:
            raise ValueError(
                f"Step ({step:g}s) > duration ({self.duration:g}s)."
            )
        self.step = step
        self.batch_size = batch_size

    def to(self, device: torch.device) -> "Inference":
        self.model.to(device)
        self.conversion.to(device)
        self.device = device
        return self

    def infer(self, chunks: torch.Tensor) -> np.ndarray | tuple[np.ndarray]:
        with torch.inference_mode():
            outputs = self.model(chunks.to(self.device))

        def __convert(output, conversion, **kwargs):
            return conversion(output).cpu().numpy()

        return _map_with_specifications(
            self.model.specifications, __convert, outputs, self.conversion
        )

    def slide(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        hook: Callable | None,
    ) -> SlidingWindowFeature | tuple[SlidingWindowFeature]:
        window_size = self.model.audio.get_num_samples(self.duration)
        step_size = round(self.step * sample_rate)
        _, num_samples = waveform.shape

        def __frames(receptive_field, specifications=None):
            if specifications.resolution == Resolution.CHUNK:
                return SlidingWindow(start=0.0, duration=self.duration, step=self.step)
            return receptive_field

        frames = _map_with_specifications(
            self.model.specifications, __frames, self.model.receptive_field
        )

        if num_samples >= window_size:
            chunks = rearrange(
                waveform.unfold(1, window_size, step_size),
                "channel chunk frame -> chunk channel frame",
            )
            num_chunks, _, _ = chunks.shape
        else:
            num_chunks = 0

        has_last_chunk = (num_samples < window_size) or (
            num_samples - window_size
        ) % step_size > 0
        if has_last_chunk:
            last_chunk = waveform[:, num_chunks * step_size :]
            _, last_window_size = last_chunk.shape
            last_pad = window_size - last_window_size
            last_chunk = F.pad(last_chunk, (0, last_pad))

        def __empty_list(**kwargs):
            return list()

        outputs = _map_with_specifications(self.model.specifications, __empty_list)

        if hook is not None:
            hook(completed=0, total=num_chunks + has_last_chunk)

        def __append_batch(output, batch_output, **kwargs):
            output.append(batch_output)

        for c in np.arange(0, num_chunks, self.batch_size):
            batch = chunks[c : c + self.batch_size]
            batch_outputs = self.infer(batch)
            _map_with_specifications(
                self.model.specifications, __append_batch, outputs, batch_outputs
            )
            if hook is not None:
                hook(completed=c + self.batch_size, total=num_chunks + has_last_chunk)

        if has_last_chunk:
            last_outputs = self.infer(last_chunk[None])
            _map_with_specifications(
                self.model.specifications, __append_batch, outputs, last_outputs
            )
            if hook is not None:
                hook(completed=num_chunks + has_last_chunk, total=num_chunks + has_last_chunk)

        def __vstack(output, **kwargs):
            return np.vstack(output)

        outputs = _map_with_specifications(
            self.model.specifications, __vstack, outputs
        )

        def __aggregate(outputs, frames, specifications=None):
            if (
                self.skip_aggregation
                or specifications.resolution == Resolution.CHUNK
                or (specifications.permutation_invariant and self.pre_aggregation_hook is None)
            ):
                frames = SlidingWindow(start=0.0, duration=self.duration, step=self.step)
                return SlidingWindowFeature(outputs, frames)

            if self.pre_aggregation_hook is not None:
                outputs = self.pre_aggregation_hook(outputs)

            aggregated = self.aggregate(
                SlidingWindowFeature(
                    outputs,
                    SlidingWindow(start=0.0, duration=self.duration, step=self.step),
                ),
                frames,
                warm_up=self.warm_up,
                hamming=True,
                missing=0.0,
            )

            if has_last_chunk:
                aggregated.data = aggregated.crop(
                    Segment(0.0, num_samples / sample_rate), mode="loose"
                )

            return aggregated

        return _map_with_specifications(
            self.model.specifications, __aggregate, outputs, frames
        )

    def __call__(
        self, file: AudioFile, hook: Callable | None = None
    ) -> SlidingWindowFeature | np.ndarray:
        fix_reproducibility(self.device)

        waveform, sample_rate = self.model.audio(file)

        if self.window == "sliding":
            return self.slide(waveform, sample_rate, hook=hook)

        outputs = self.infer(waveform[None])

        def __first_sample(outputs, **kwargs):
            return outputs[0]

        return _map_with_specifications(
            self.model.specifications, __first_sample, outputs
        )

    def crop(
        self,
        file: AudioFile,
        chunk: Segment | list[Segment],
        hook: Callable | None = None,
    ) -> SlidingWindowFeature | np.ndarray:
        fix_reproducibility(self.device)

        if self.window == "sliding":
            if not isinstance(chunk, Segment):
                start = min(c.start for c in chunk)
                end = max(c.end for c in chunk)
                chunk = Segment(start=start, end=end)

            waveform, sample_rate = self.model.audio.crop(file, chunk)
            outputs = self.slide(waveform, sample_rate, hook=hook)

            def __shift(output, **kwargs):
                frames = output.sliding_window
                shifted_frames = SlidingWindow(
                    start=chunk.start, duration=frames.duration, step=frames.step
                )
                return SlidingWindowFeature(output.data, shifted_frames)

            return _map_with_specifications(self.model.specifications, __shift, outputs)

        if isinstance(chunk, Segment):
            waveform, sample_rate = self.model.audio.crop(file, chunk)
        else:
            waveform = torch.cat(
                [self.model.audio.crop(file, c)[0] for c in chunk], dim=1
            )

        outputs = self.infer(waveform[None])

        def __first_sample(outputs, **kwargs):
            return outputs[0]

        return _map_with_specifications(
            self.model.specifications, __first_sample, outputs
        )

    @staticmethod
    def aggregate(
        scores: SlidingWindowFeature,
        frames: SlidingWindow,
        warm_up: tuple[float, float] = (0.0, 0.0),
        epsilon: float = 1e-12,
        hamming: bool = False,
        missing: float = np.nan,
        skip_average: bool = False,
    ) -> SlidingWindowFeature:
        num_chunks, num_frames_per_chunk, num_classes = scores.data.shape

        chunks = scores.sliding_window
        frames = SlidingWindow(
            start=chunks.start, duration=frames.duration, step=frames.step,
        )

        hamming_window = (
            np.hamming(num_frames_per_chunk).reshape(-1, 1)
            if hamming
            else np.ones((num_frames_per_chunk, 1))
        )

        warm_up_window = np.ones((num_frames_per_chunk, 1))
        warm_up_left = round(
            warm_up[0] / scores.sliding_window.duration * num_frames_per_chunk
        )
        warm_up_window[:warm_up_left] = epsilon
        warm_up_right = round(
            warm_up[1] / scores.sliding_window.duration * num_frames_per_chunk
        )
        warm_up_window[num_frames_per_chunk - warm_up_right :] = epsilon

        num_frames = (
            frames.closest_frame(
                scores.sliding_window.start
                + scores.sliding_window.duration
                + (num_chunks - 1) * scores.sliding_window.step
                + 0.5 * frames.duration
            )
            + 1
        )
        aggregated_output = np.zeros((num_frames, num_classes), dtype=np.float32)
        overlapping_chunk_count = np.zeros((num_frames, num_classes), dtype=np.float32)
        aggregated_mask = np.zeros((num_frames, num_classes), dtype=np.float32)

        for chunk, score in scores:
            mask = 1 - np.isnan(score)
            np.nan_to_num(score, copy=False, nan=0.0)

            start_frame = frames.closest_frame(chunk.start + 0.5 * frames.duration)

            aggregated_output[start_frame : start_frame + num_frames_per_chunk] += (
                score * mask * hamming_window * warm_up_window
            )
            overlapping_chunk_count[start_frame : start_frame + num_frames_per_chunk] += (
                mask * hamming_window * warm_up_window
            )
            aggregated_mask[start_frame : start_frame + num_frames_per_chunk] = np.maximum(
                aggregated_mask[start_frame : start_frame + num_frames_per_chunk], mask,
            )

        if skip_average:
            average = aggregated_output
        else:
            average = aggregated_output / np.maximum(overlapping_chunk_count, epsilon)

        average[aggregated_mask == 0.0] = missing

        return SlidingWindowFeature(average, frames)
