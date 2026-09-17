"""Custom Ultralytics modules used by the OGSOD OBB experiments."""

from .eca import ECA, register_custom_modules

__all__ = ["ECA", "register_custom_modules"]
