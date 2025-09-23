# options.py helper functions
from __future__ import annotations
from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf

def _p(
    x: Optional[str] | Optional[Path],
    root: Optional[Path] = None
) -> Optional[Path]:

    if x is None:
        return None
    x = Path(x)
    if root is not None and not x.is_absolute():
        return (root / x).resolve()
    return x.resolve()

def _structured_merge_to_obj(cls, section) -> object:
    """
    Merge a YAML section onto a structured
    config made from the dataclass `cls`,
    then return a dataclass instance.
    """
    base = OmegaConf.structured(cls)
    merged = OmegaConf.merge(base, section or {})
    return OmegaConf.to_object(merged)