"""Run: python check_spaces.py  -> shows which Hugging Face Spaces are alive right now."""
from huggingface_hub import HfApi
import pipeline
for sp in [pipeline.CONTROLNET_SPACE] + pipeline.TRELLIS_SPACES:
    try:
        print(f"{sp:45s} {HfApi().get_space_runtime(sp).stage}")
    except Exception as e:
        print(f"{sp:45s} ERROR {str(e)[:80]}")
