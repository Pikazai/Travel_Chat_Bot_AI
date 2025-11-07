"""
Configuration module for Travel Chat Bot AI.
Handles environment variables, constants, and application settings.
"""

try:
    from .settings import Settings, get_settings
except ImportError:
    # Handle relative import issues
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]

