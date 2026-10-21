"""Quick test: load Qwen2.5-7B on MPS and generate text."""
# Stub _lzma if missing (e.g. pyenv Python built without liblzma) so transformers/torchvision can import
import sys

if "_lzma" not in sys.modules:
    try:
        import _lzma  # noqa: F401
    except ModuleNotFoundError:
        import types

        _stub = types.ModuleType("_lzma")
        _stub.LZMAError = type("LZMAError", (Exception,), {})
        for name in (
            "CHECK_NONE", "CHECK_CRC32", "CHECK_CRC64", "CHECK_SHA256",
            "CHECK_ID_MAX", "CHECK_UNKNOWN",
            "FORMAT_AUTO", "FORMAT_XZ", "FORMAT_ALONE", "FORMAT_RAW",
            "FILTER_LZMA1", "FILTER_LZMA2", "FILTER_DELTA", "FILTER_X86",
            "FILTER_IA64", "FILTER_ARM", "FILTER_ARMTHUMB", "FILTER_POWERPC", "FILTER_SPARC",
            "MF_HC3", "MF_HC4", "MF_BT2", "MF_BT3", "MF_BT4",
            "MODE_FAST", "MODE_NORMAL", "PRESET_DEFAULT", "PRESET_EXTREME",
        ):
            setattr(_stub, name, 0)
        _stub._encode_filter_properties = lambda *a, **k: None
        _stub._decode_filter_properties = lambda *a, **k: None
        _stub.LZMACompressor = type("LZMACompressor", (), {})
        _stub.LZMADecompressor = type("LZMADecompressor", (), {})
        _stub.open = lambda *a, **k: None
        _stub.compress = lambda x: x
        _stub.decompress = lambda x: x
        _stub.is_check_supported = lambda x: True
        sys.modules["_lzma"] = _stub

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    device_map="mps",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

inputs = tokenizer("你好", return_tensors="pt").to("mps")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
print("Done.")
