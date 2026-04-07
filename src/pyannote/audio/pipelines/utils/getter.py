"""Model getter - minimal version for local loading only."""

from pathlib import Path
from typing import Mapping

from pyannote.audio.core.model import Model

PipelineModel = Model | str | Mapping


def get_model(
    model: PipelineModel,
    token: str | None = None,
    cache_dir: Path | str | None = None,
) -> Model:
    """Load pretrained model and set it into eval mode."""

    if isinstance(model, Model):
        pass
    elif isinstance(model, str):
        model = Model.from_pretrained(model, strict=False)
    elif isinstance(model, Mapping):
        model = Model.from_pretrained(**model)
    else:
        raise TypeError(f"Unsupported type ({type(model)}) for loading model.")

    model.eval()
    return model
