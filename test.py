import sys
import time
import os

print("--- DETECTIVE MODE STARTED ---")
print(f"Python: {sys.version.split()[0]}")
print("step 1: importing basic libs...")
import warnings
import pathlib

time.sleep(0.5)

print("step 2: importing torch...")
import torch

print(f"   -> CUDA Available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   -> Version: {torch.version.cuda}")
time.sleep(0.5)

print("step 3: importing torchvision...")
import torchvision

time.sleep(0.5)

print("step 4: importing PIL & opencv...")
import PIL
import cv2

time.sleep(0.5)

print("step 5: importing sqlmodel...")
import sqlmodel

time.sleep(0.5)

# --- THE USUAL SUSPECTS ---

print("step 6: importing triton...")
try:
    import triton

    print(f"   -> Triton imported successfully")
except ImportError:
    print("   -> Triton NOT found")
except Exception as e:
    print(f"   -> Triton error: {e}")
time.sleep(1)

print("step 7: importing xformers...")
try:
    import xformers
    import xformers.ops

    print(f"   -> xFormers imported. Ops available: {xformers.ops.is_available()}")
except ImportError:
    print("   -> xFormers NOT found")
except Exception as e:
    print(f"   -> xFormers error: {e}")
time.sleep(1)

print("step 8: importing transformers (The heavy hitter)...")
# Transformers often lazy-loads other things, so we force a check
try:
    import transformers

    print(f"   -> Transformers version: {transformers.__version__}")

    # Force it to look for quantization libs
    from transformers import BitsAndBytesConfig

    print("   -> BitsAndBytesConfig imported")
except ImportError:
    print("   -> Transformers or sub-module missing")
except Exception as e:
    print(f"   -> Transformers error: {e}")
time.sleep(1)

print("step 9: checking for bitsandbytes directly...")
try:
    import bitsandbytes

    print("   -> bitsandbytes imported")
except ImportError:
    print("   -> bitsandbytes NOT installed (This is good on Windows usually)")
except Exception as e:
    print(f"   -> bitsandbytes error: {e}")

print("--- INVESTIGATION COMPLETE ---")