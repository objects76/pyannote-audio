"""Signal processing - Binarize only"""

from functools import singledispatch
from itertools import zip_longest

import einops
import numpy as np
from pyannote.core import Annotation, Segment, SlidingWindowFeature
from pyannote.core.utils.generators import pairwise, string_generator


@singledispatch
def binarize(scores, onset=0.5, offset=None, initial_state=None):
    raise NotImplementedError("scores must be np.ndarray or SlidingWindowFeature")


@binarize.register
def binarize_ndarray(
    scores: np.ndarray, onset=0.5, offset=None, initial_state=None,
):
    offset = offset or onset
    batch_size, num_frames = scores.shape
    scores = np.nan_to_num(scores)

    if initial_state is None:
        initial_state = scores[:, 0] >= 0.5 * (onset + offset)
    elif isinstance(initial_state, bool):
        initial_state = initial_state * np.ones((batch_size,), dtype=bool)

    initial_state = np.tile(initial_state, (num_frames, 1)).T

    on = scores > onset
    off_or_on = (scores < offset) | on

    well_defined_idx = np.array(
        list(zip_longest(*[np.nonzero(oon)[0] for oon in off_or_on], fillvalue=-1))
    ).T

    if not well_defined_idx.size:
        return np.zeros_like(scores, dtype=bool) | initial_state

    same_as = np.cumsum(off_or_on, axis=1)
    samples = np.tile(np.arange(batch_size), (num_frames, 1)).T

    return np.where(
        same_as, on[samples, well_defined_idx[samples, same_as - 1]], initial_state
    )


@binarize.register
def binarize_swf(
    scores: SlidingWindowFeature, onset=0.5, offset=None, initial_state=None,
):
    offset = offset or onset

    if scores.data.ndim == 2:
        num_frames, num_classes = scores.data.shape
        data = einops.rearrange(scores.data, "f k -> k f", f=num_frames, k=num_classes)
        binarized = binarize(data, onset=onset, offset=offset, initial_state=initial_state)
        return SlidingWindowFeature(
            1.0 * einops.rearrange(binarized, "k f -> f k", f=num_frames, k=num_classes),
            scores.sliding_window,
        )

    elif scores.data.ndim == 3:
        num_chunks, num_frames, num_classes = scores.data.shape
        data = einops.rearrange(
            scores.data, "c f k -> (c k) f", c=num_chunks, f=num_frames, k=num_classes
        )
        binarized = binarize(data, onset=onset, offset=offset, initial_state=initial_state)
        return SlidingWindowFeature(
            1.0 * einops.rearrange(
                binarized, "(c k) f -> c f k", c=num_chunks, f=num_frames, k=num_classes
            ),
            scores.sliding_window,
        )

    else:
        raise ValueError("Shape must be (num_chunks, num_frames, num_classes) or (num_frames, num_classes).")


class Binarize:
    """Binarize detection scores using hysteresis thresholding."""

    def __init__(
        self,
        onset: float = 0.5,
        offset: float | None = None,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
        pad_onset: float = 0.0,
        pad_offset: float = 0.0,
    ):
        super().__init__()
        self.onset = onset
        self.offset = offset or onset
        self.pad_onset = pad_onset
        self.pad_offset = pad_offset
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off

    def __call__(self, scores: SlidingWindowFeature) -> Annotation:
        num_frames, num_classes = scores.data.shape
        frames = scores.sliding_window
        timestamps = [frames[i].middle for i in range(num_frames)]

        active = Annotation()
        track_generator = string_generator()

        for k, k_scores in enumerate(scores.data.T):
            label = k if scores.labels is None else scores.labels[k]
            track = next(track_generator)

            start = timestamps[0]
            is_active = k_scores[0] > self.onset

            for t, y in zip(timestamps[1:], k_scores[1:]):
                if is_active:
                    if y < self.offset:
                        region = Segment(start - self.pad_onset, t + self.pad_offset)
                        active[region, track] = label
                        start = t
                        is_active = False
                else:
                    if y > self.onset:
                        start = t
                        is_active = True

            if is_active:
                region = Segment(start - self.pad_onset, t + self.pad_offset)
                active[region, track] = label

        if self.pad_offset > 0.0 or self.pad_onset > 0.0 or self.min_duration_off > 0.0:
            active = active.support(collar=self.min_duration_off)

        if self.min_duration_on > 0:
            for segment, track in list(active.itertracks()):
                if segment.duration < self.min_duration_on:
                    del active[segment, track]

        return active
