from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path

import scipy.special
import torch
import torch.nn as nn
from pyannote.audio.core.io import Audio
from pyannote.core import SlidingWindow


class Problem(Enum):
    BINARY_CLASSIFICATION = 0
    MONO_LABEL_CLASSIFICATION = 1
    MULTI_LABEL_CLASSIFICATION = 2
    REPRESENTATION = 3
    REGRESSION = 4


class Resolution(Enum):
    FRAME = 1
    CHUNK = 2


class UnknownSpecificationsError(Exception):
    pass


@dataclass
class Specifications:
    problem: Problem
    resolution: Resolution
    duration: float
    min_duration: float | None = None
    warm_up: tuple[float, float] | None = (0.0, 0.0)
    classes: list[str] | None = None
    powerset_max_classes: int | None = None
    permutation_invariant: bool = False

    @cached_property
    def powerset(self) -> bool:
        if self.powerset_max_classes is None:
            return False
        return True

    @cached_property
    def num_powerset_classes(self) -> int:
        return int(
            sum(
                scipy.special.binom(len(self.classes), i)
                for i in range(0, self.powerset_max_classes + 1)
            )
        )

    def __len__(self):
        return 1

    def __iter__(self):
        yield self


@dataclass
class Output:
    num_frames: int
    dimension: int
    frames: SlidingWindow


class _HParams:
    """Simple namespace to replace lightning's hparams."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)


class Model(nn.Module):
    """Base model (inference-only, no lightning dependency)"""

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task=None,
    ):
        super().__init__()

        assert num_channels == 1, "Only mono audio is supported (num_channels = 1)"

        self.hparams = _HParams(sample_rate=sample_rate, num_channels=num_channels)
        self.audio = Audio(sample_rate=sample_rate, mono="downmix")
        self._specifications = None

    def save_hyperparameters(self, *args):
        """Store constructor args in self.hparams (replaces lightning's method)."""
        import inspect

        frame = inspect.currentframe().f_back
        local_vars = frame.f_locals
        for name in args:
            setattr(self.hparams, name, local_vars[name])

    @property
    def specifications(self) -> Specifications | tuple[Specifications]:
        if self._specifications is None:
            raise UnknownSpecificationsError(
                "Model specifications are not available."
            )
        return self._specifications

    @specifications.setter
    def specifications(self, specifications):
        self._specifications = specifications

    @specifications.deleter
    def specifications(self):
        self._specifications = None

    @cached_property
    def receptive_field(self) -> SlidingWindow:
        receptive_field_size = self.receptive_field_size(num_frames=1)
        receptive_field_step = (
            self.receptive_field_size(num_frames=2) - receptive_field_size
        )
        receptive_field_start = (
            self.receptive_field_center(frame=0) - (receptive_field_size - 1) / 2
        )
        return SlidingWindow(
            start=receptive_field_start / self.hparams.sample_rate,
            duration=receptive_field_size / self.hparams.sample_rate,
            step=receptive_field_step / self.hparams.sample_rate,
        )

    def build(self):
        pass

    def forward(self, waveforms: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def default_activation(self) -> nn.Module:
        specs = self.specifications
        if isinstance(specs, tuple):
            raise ValueError("Multi-task not supported in minimal VAD.")
        if specs.problem == Problem.BINARY_CLASSIFICATION:
            return nn.Sigmoid()
        elif specs.problem == Problem.MONO_LABEL_CLASSIFICATION:
            return nn.LogSoftmax(dim=-1)
        elif specs.problem == Problem.MULTI_LABEL_CLASSIFICATION:
            return nn.Sigmoid()
        else:
            raise NotImplementedError

    # --- Hardcoded model config (from config.yaml / checkpoint metadata) ---
    _HARDCODED_HPARAMS = {
        "sample_rate": 16000,
        "num_channels": 1,
        "sincnet": {"stride": 10, "sample_rate": 16000},
        "lstm": {
            "hidden_size": 128,
            "num_layers": 4,
            "bidirectional": True,
            "monolithic": True,
            "dropout": 0.5,
            "batch_first": True,
        },
        "linear": {"hidden_size": 128, "num_layers": 2},
    }

    _HARDCODED_SPECIFICATIONS = Specifications(
        problem=Problem.MONO_LABEL_CLASSIFICATION,
        resolution=Resolution.FRAME,
        duration=10.0,
        min_duration=None,
        warm_up=(0.0, 0.0),
        classes=["speaker#1", "speaker#2", "speaker#3"],
        powerset_max_classes=2,
        permutation_invariant=True,
    )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: Path | str,
        map_location=None,
        strict: bool = True,
        **kwargs,
    ) -> "Model":
        """Load pretrained PyanNet model from local pytorch_model.bin checkpoint."""
        from pyannote.audio.models.segmentation.PyanNet import PyanNet

        checkpoint = Path(checkpoint)

        if checkpoint.is_dir():
            checkpoint = checkpoint / "pytorch_model.bin"

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        if map_location is None:
            map_location = lambda storage, loc: storage

        loaded = torch.load(checkpoint, map_location=map_location, weights_only=False)

        # hardcoded PyanNet instantiation (no dynamic import, no config.yaml)
        hparams = dict(cls._HARDCODED_HPARAMS)
        hparams.update(kwargs)
        model = PyanNet(**hparams)

        model.specifications = cls._HARDCODED_SPECIFICATIONS
        model.build()

        state_dict = loaded["state_dict"]
        model.load_state_dict(state_dict, strict=strict)

        return model
