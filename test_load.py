import sys
import logging
import torch

from transformers import AutoTokenizer
from adapters import AutoAdapterModel

logging.basicConfig(level=logging.INFO)

base_path = "tmp/matrix_cache/specter2_base"
adapter_path = "tmp/matrix_cache/specter2_adapter"

print("Loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
print("Loading base model...", flush=True)
mdl = AutoAdapterModel.from_pretrained(base_path, local_files_only=True)

print("Loading adapter...", flush=True)
mdl.load_adapter(adapter_path, load_as="adhoc_query", set_active=True)

print("Moving to CPU...", flush=True)
tok, mdl = tok, mdl.to("cpu").eval()

print("Finish!", flush=True)
