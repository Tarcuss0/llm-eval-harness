"""evalkit — прогон наборов, пороги приёмки и поиск регрессий для LLM-задач."""

from .core import Case, Config, Prediction, RunResult, load_cases
from .compare import compare
from .runner import run, save, load

__all__ = ["Case", "Config", "Prediction", "RunResult", "load_cases", "run", "save", "load", "compare"]
__version__ = "0.1.0"
