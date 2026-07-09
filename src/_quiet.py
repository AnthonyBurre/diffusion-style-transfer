"""Silence known-harmless upstream warnings.

Imported first by both entrypoints (``src.app`` and ``src.cli``), before the
pipeline import, so the filters are installed before the noisy libraries load.
"""
import warnings

# We only use canny + depth conditioning, dont need mediapipe
warnings.filterwarnings("ignore", message="The module 'mediapipe' is not installed")
# Emitted from inside controlnet_aux, await an upstream release
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")
warnings.filterwarnings("ignore", message="Importing from timm.models.registry is deprecated")
# Internal to the library
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_")
# Emitted from inside diffusers' load_ip_adapter via huggingface_hub
warnings.filterwarnings("ignore", message="The `local_dir_use_symlinks` argument is deprecated")
