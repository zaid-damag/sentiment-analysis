import importlib

mods = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("tokenizers", "tokenizers"),
    ("datasets", "datasets"),
    ("gradio", "gradio"),
    ("zmq", "zmq"),
    ("huggingface_hub", "huggingface_hub"),
    ("hf_xet", "hf_xet"),
]

all_ok = True
for label, mod in mods:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "N/A")
        print(f"✓ {label} import OK  (version: {ver})")
    except Exception as e:
        all_ok = False
        print(f"✗ {label} FAILED: {e}")

# تفاصيل إضافية لـ torch/vision
try:
    import torch, torchvision
    print("Torch:", torch.__version__, "| TorchVision:", torchvision.__version__)
except Exception as e:
    print("Torch/TorchVision extra info error:", e)

print("\nAll OK" if all_ok else "\nSome imports failed")
