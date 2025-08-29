"""
Tessera: Next-Generation Deep Learning Programming Model

A revolutionary deep learning framework that treats numerical precision, 
data movement, parallelism, and correctness as first-class semantic objects.
"""

__version__ = "0.1.0"
__author__ = "Tessera Team"

# Core imports
from . import core
from . import nn
from . import compiler
from . import runtime
from . import utils

# Convenient aliases
from .core import Tensor, Module
from .compiler import compile

__all__ = [
    "core",
    "nn", 
    "compiler",
    "runtime",
    "utils",
    "Tensor",
    "Module",
    "compile"
]
