"""Chalao: python check_setup.py   (ecotex_app folder ke andar). Batata hai model kyun load nahi ho raha."""
import os
import sys

print("Python:", sys.version.split()[0])
try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
except Exception as e:
    print("TensorFlow ERROR:", e)
here = os.path.dirname(os.path.abspath(__file__))
md = os.path.join(here, "model")
print("Model folder:", md, "| exists:", os.path.isdir(md))
print("Files inside:", os.listdir(md) if os.path.isdir(md) else None)
import app  # noqa: E402  (ye model load karne ki koshish karta hai)
print("RESULT:", app.MODEL_STATUS)
