"""
Simplified train.py file that doesn't require Dora.
"""

import dataclasses
import logging
import os
from pathlib import Path
import sys
import typing as tp

import torch

logger = logging.getLogger(__name__)


class _DummyObject:
    """Dummy object to use as a placeholder for Dora."""
    
    def __init__(self):
        self.dir = Path("/tmp/dora/xps")
        os.makedirs(self.dir, exist_ok=True)


@dataclasses.dataclass
class _DummyMain:
    """Dummy object that mimics the Dora main object."""
    
    dora: _DummyObject = dataclasses.field(default_factory=_DummyObject)


# Create a dummy main object
main = _DummyMain()


def init(argv: tp.Optional[tp.List[str]] = None) -> _DummyMain:
    """Initialize the main object."""
    return main 