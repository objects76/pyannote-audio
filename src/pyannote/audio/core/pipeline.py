"""Pipeline base class - minimal version without pyannote.pipeline dependency"""

import warnings
from collections import OrderedDict
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
from pyannote.audio.core.inference import BaseInference
from pyannote.audio.core.io import Audio, AudioFile
from pyannote.audio.core.model import Model
from pyannote.audio.utils.reproducibility import fix_reproducibility


class Pipeline:
    """Minimal pipeline base class (inference only, no optuna/pyannote.pipeline)."""

    def __init__(self):
        self._models: dict[str, Model] = OrderedDict()
        self._inferences: dict[str, BaseInference] = OrderedDict()
        self._instantiated = False
        self.device = torch.device("cpu")

    @property
    def instantiated(self) -> bool:
        return self._instantiated

    def instantiate(self, parameters: dict):
        for name, value in parameters.items():
            setattr(self, name, value)
        self._instantiated = True
        self.initialize()
        return self

    def initialize(self):
        """Override in subclasses to set up internal state after instantiate()."""
        pass

    def __getattr__(self, name):
        if "_models" in self.__dict__:
            _models = self.__dict__["_models"]
            if name in _models:
                return _models[name]

        if "_inferences" in self.__dict__:
            _inferences = self.__dict__["_inferences"]
            if name in _inferences:
                return _inferences[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        _models = self.__dict__.get("_models")
        _inferences = self.__dict__.get("_inferences")

        if isinstance(value, nn.Module) and _models is not None:
            if name in self.__dict__:
                del self.__dict__[name]
            _models[name] = value
            return

        if isinstance(value, BaseInference) and _inferences is not None:
            if name in self.__dict__:
                del self.__dict__[name]
            _inferences[name] = value
            return

        super().__setattr__(name, value)

    @staticmethod
    def setup_hook(file: AudioFile, hook: Callable | None = None) -> Callable:
        def noop(*args, **kwargs):
            return

        return partial(hook or noop, file=file)

    def default_parameters(self):
        raise NotImplementedError()

    def apply(self, file: AudioFile, **kwargs):
        raise NotImplementedError()

    def __call__(self, file: AudioFile, **kwargs):
        fix_reproducibility(self.device)

        if not self.instantiated:
            try:
                default_parameters = self.default_parameters()
            except NotImplementedError:
                raise RuntimeError(
                    "Pipeline must be instantiated with `pipeline.instantiate(parameters)` first."
                )
            self.instantiate(default_parameters)
            warnings.warn(f"Pipeline auto-instantiated with {default_parameters}.")

        file = Audio.validate_file(file)
        return self.apply(file, **kwargs)

    def to(self, device: torch.device) -> "Pipeline":
        if not isinstance(device, torch.device):
            raise TypeError(f"`device` must be torch.device, got `{type(device).__name__}`")

        for _, model in self._models.items():
            model.to(device)

        for _, inference in self._inferences.items():
            inference.to(device)

        self.device = device
        return self
