"""Optional local VLM backend for the figure-enrichment crawler.

Offline preprocessing only: it is never imported by the classifier
(patent_mpri) and no image tensors/embeddings leave this module; its sole
output is validated text.

No model weights live in the repository. Weights are downloaded by Hugging
Face `transformers` into the user's HF cache on first use. Heavy dependencies
are imported lazily, so importing this file needs neither torch nor
transformers. See requirements-vlm-local.txt.

Interface (shared with the API backend): `.model`, `.model_version`,
`.config_hash`, `.describe(image_bytes, mime, figure_id) -> (dict, usage)`.
Prompt text, the output parser and the preprocessing version are injected by
the crawler so both backends use identical prompts and validation.
"""
from __future__ import annotations

import hashlib
import io
import json


class LocalVlmError(Exception):
    def __init__(self, msg, permanent=False):
        super().__init__(msg)
        self.permanent = permanent


class LocalTransformersVLM:
    """Any `transformers` image-text-to-text model with a chat template
    (e.g. Qwen2.5-VL). Greedy decoding (do_sample=False) for reproducibility;
    exact bit-reproducibility still depends on hardware/library versions."""

    def __init__(self, model_id, system_prompt, user_prompt, parser, error_cls,
                 prompt_version, image_prep_version, revision=None,
                 dtype="bfloat16", device="auto", load_in_4bit=False,
                 max_new_tokens=1200, max_retries=1, loader=None):
        self.model, self.revision = model_id, revision
        self.system_prompt, self.user_prompt = system_prompt, user_prompt
        self.parser, self.error_cls = parser, error_cls
        self.prompt_version, self.image_prep_version = prompt_version, image_prep_version
        self.dtype, self.device, self.load_in_4bit = dtype, device, load_in_4bit
        self.max_new_tokens, self.max_retries = max_new_tokens, max_retries
        self._loader, self._loaded = loader, None
        self.model_version = revision or ""

    @property
    def config_hash(self):
        cfg = {"backend": "local-transformers", "model": self.model,
               "revision": self.revision, "dtype": self.dtype,
               "4bit": self.load_in_4bit, "prompt": self.prompt_version,
               "max_new_tokens": self.max_new_tokens, "greedy": True,
               "image_prep": self.image_prep_version}
        return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()
                              ).hexdigest()[:16]

    def _load(self):
        if self._loaded:
            return self._loaded
        if self._loader:                      # test seam
            self._loaded = self._loader()
            return self._loaded
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:
            raise self.error_cls(
                "local VLM dependencies missing (pip install -r "
                f"requirements-vlm-local.txt): {e}", permanent=True)
        kw = {"revision": self.revision} if self.revision else {}
        mkw = dict(kw, device_map=self.device)
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            mkw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            mkw["torch_dtype"] = getattr(torch, self.dtype)
        try:
            proc = AutoProcessor.from_pretrained(self.model, **kw)
            mdl = AutoModelForImageTextToText.from_pretrained(self.model, **mkw)
        except Exception as e:
            raise self.error_cls(f"could not load {self.model}: {type(e).__name__}: {e}",
                                 permanent=True)
        commit = getattr(getattr(mdl, "config", None), "_commit_hash", None)
        if commit and not self.revision:
            self.model_version = commit       # record the resolved revision
        self._loaded = (mdl, proc)
        return self._loaded

    def describe(self, image_bytes, mime, figure_id):
        from PIL import Image
        mdl, proc = self._load()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": self.user_prompt.format(figure_id=figure_id)}]}]
        last = "unknown"
        for _ in range(self.max_retries + 1):
            try:
                text, n_in, n_out = self._generate(mdl, proc, messages, img)
            except self.error_cls:
                raise
            except Exception as e:            # e.g. CUDA OOM
                last = f"{type(e).__name__}: {str(e)[:120]}"
                if "OutOfMemory" in type(e).__name__:
                    raise self.error_cls(f"GPU out of memory: {last}", permanent=True)
                continue
            try:
                return self.parser(text), {"prompt_tokens": n_in,
                                           "completion_tokens": n_out}
            except ValueError as e:
                last = f"invalid_output: {e}"
        raise self.error_cls(f"gave up after {self.max_retries + 1} attempts: {last}")

    def _generate(self, mdl, proc, messages, img):
        prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=[prompt], images=[img], return_tensors="pt").to(mdl.device)
        n_in = int(inputs["input_ids"].shape[-1])
        out = mdl.generate(**inputs, max_new_tokens=self.max_new_tokens,
                           do_sample=False)
        new = out[:, n_in:]
        text = proc.batch_decode(new, skip_special_tokens=True)[0]
        return text, n_in, int(new.shape[-1])
