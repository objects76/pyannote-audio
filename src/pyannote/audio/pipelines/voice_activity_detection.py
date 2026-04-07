"""Voice activity detection pipeline - minimal inference-only version."""

from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
from pyannote.core import Annotation, SlidingWindowFeature

from pyannote.audio import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import PipelineModel, get_model
from pyannote.audio.utils.signal import Binarize


class VoiceActivityDetection(Pipeline):
    """Voice activity detection pipeline

    Parameters
    ----------
    segmentation : Model, str, or dict
        Pretrained segmentation (or VAD) model.
    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        fscore: bool = False,
        token: str | None = None,
        cache_dir: Path | str | None = None,
        **inference_kwargs,
    ):
        super().__init__()

        self.segmentation = segmentation
        self.fscore = fscore

        model = get_model(segmentation, token=token, cache_dir=cache_dir)

        inference_kwargs["pre_aggregation_hook"] = lambda scores: np.max(
            scores, axis=-1, keepdims=True
        )
        self._segmentation = Inference(model, **inference_kwargs)

    def default_parameters(self):
        return {
            "onset": 0.5,
            "offset": 0.5,
            "min_duration_on": 0.0,
            "min_duration_off": 0.0,
        }

    def classes(self):
        return ["SPEECH"]

    def initialize(self):
        self._binarize = Binarize(
            onset=self.onset,
            offset=self.offset,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )

    def apply(self, file: AudioFile, hook: Callable | None = None) -> Annotation:
        hook = self.setup_hook(file, hook=hook)

        segmentations: SlidingWindowFeature = self._segmentation(
            file, hook=partial(hook, "segmentation", None)
        )
        hook("segmentation", segmentations)

        speech: Annotation = self._binarize(segmentations)
        speech.uri = file["uri"]
        return speech.rename_labels({label: "SPEECH" for label in speech.labels()})
