"""
LLM backend abstraction: supports OpenAI GPT API, local Qwen, and vLLM server.
Selected via environment variable LLM_BACKEND: openai (default) | qwen | vllm.
When using Qwen, set QWEN_MODEL to choose the model; default is Qwen3-4B
(HuggingFace: Qwen/Qwen3-4B-Instruct-2507).
Qwen3 requires transformers >= 4.51.

Download and memory options (effective only when LLM_BACKEND=qwen):
  QWEN_CACHE_DIR       Model cache directory (blobs/snapshots), default ~/.cache/huggingface/hub.
  QWEN_MAX_NEW_TOKENS  Maximum generated tokens, default 2048. Lower values save memory.
  QWEN_MAX_GPU_MEMORY  Per-GPU memory cap, e.g. "20GiB" (requires accelerate).
  PYTORCH_CUDA_ALLOC_CONF  PyTorch allocator config, e.g. expandable_segments:True to reduce fragmentation.

Main GPU memory allocation points in this file:
  1. _get_qwen_manager: from_pretrained + device_map / .to(device) for model weights.
  2. _get_qwen_manager: optional max_memory control when QWEN_MAX_GPU_MEMORY is set.
  3. _qwen_generate: model_inputs.to(model.device) for prompt tensors.
  4. _qwen_generate: model.generate() for KV cache and forward activations.
"""
import os
import contextlib
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

# Default backend: openai when not set.
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai").lower().strip()
# Default Qwen3-4B Instruct (https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
# Model cache directory (blobs/snapshots), default local cache.
QWEN_CACHE_DIR = os.getenv("QWEN_CACHE_DIR", os.path.expanduser("~/.cache/huggingface/hub")).rstrip("/")
# vLLM OpenAI-compatible endpoint settings.
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_ENABLE_THINKING = os.getenv("VLLM_ENABLE_THINKING", "1").strip().lower() in {"1", "true", "yes", "on"}

def _cache_dir_with_local_locks():
    """Return cache_dir for HuggingFace (default local ~/.cache/huggingface/hub)."""
    if not QWEN_CACHE_DIR:
        return os.path.expanduser("~/.cache/huggingface/hub")
    cache_root = os.path.abspath(os.path.expanduser(QWEN_CACHE_DIR))
    try:
        os.makedirs(cache_root, mode=0o755, exist_ok=True)
    except (OSError, PermissionError):
        return os.path.expanduser("~/.cache/huggingface/hub")
    return cache_root


def _apply_hf_filelock_workaround():
    """
    Run before snapshot_download: replace FileLock used by huggingface_hub with a no-op lock
    to avoid filelock/huggingface_hub compatibility issues.
    """
    try:
        import huggingface_hub.utils._fixes as hf_fixes
    except ImportError:
        return
    if getattr(hf_fixes, "_llm_backend_lock_patched", False):
        return
    # No-op lock: accepts arbitrary args (including mode=); with/.acquire()/.release() do nothing.
    class _NoOpFileLock:
        def __init__(self, *args, **kwargs):
            pass
        def acquire(self, *args, **kwargs):
            pass
        def release(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    hf_fixes.FileLock = _NoOpFileLock
    hf_fixes._llm_backend_lock_patched = True


def _hub_id_to_local_path(model_id: str, cache_dir: str) -> str:
    """
    Download a Hub model_id into cache_dir via snapshot_download and return local snapshot path.
    This avoids relative-path ambiguity and filelock compatibility issues.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("Downloading from Hub requires: pip install huggingface_hub")
    _apply_hf_filelock_workaround()
    path = snapshot_download(model_id, cache_dir=cache_dir, local_files_only=False)
    return path
# Maximum new tokens during generation.
QWEN_MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "2048"))
# Per-GPU memory cap (effective with device_map and accelerate), e.g. "20GiB" or "18GB".
QWEN_MAX_GPU_MEMORY = os.getenv("QWEN_MAX_GPU_MEMORY", "").strip() or None
# Qwen3 default think-end token id (</think>) used to separate reasoning and final answer.
QWEN_THINK_END_TOKEN_ID = int(os.getenv("QWEN_THINK_END_TOKEN_ID", "151668"))

# Local Qwen singleton.
_qwen_manager = None
_qwen_tokenizer = None
_outlines_model = None
_outlines_json_generators: Dict[str, Any] = {}
_qwen_json_trace_seq = 0


def _build_messages(prompt: str, chat_history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    if chat_history:
        return list(chat_history) + [{"role": "user", "content": prompt}]
    return [{"role": "user", "content": prompt}]


def _vllm_chat_create(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    guided_json_schema: Optional[Dict[str, Any]] = None,
):
    try:
        import openai
    except ImportError:
        raise ImportError("vLLM backend requires: pip install openai")
    client = openai.OpenAI(api_key=VLLM_API_KEY, base_url=VLLM_BASE_URL)
    # vLLM accepts OpenAI-compatible requests; extra_body carries guided decoding.
    extra_body: Dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": bool(VLLM_ENABLE_THINKING)},
    }
    if guided_json_schema is not None:
        extra_body["guided_json"] = guided_json_schema
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        extra_body=extra_body,
    )


def _vllm_generate(messages: List[Dict[str, str]], model: str, temperature: float = 0.0) -> str:
    response = _vllm_chat_create(model=model, messages=messages, temperature=temperature)
    msg = response.choices[0].message
    # For reasoning models, vLLM puts final answer in content and reasoning in reasoning.
    return (getattr(msg, "content", None) or "").strip()


def _vllm_generate_json_schema(messages: List[Dict[str, str]], model: str, json_schema: Dict[str, Any]) -> Any:
    response = _vllm_chat_create(
        model=model,
        messages=messages,
        temperature=0.0,
        guided_json_schema=json_schema,
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        return {}
    try:
        return json.loads(content)
    except Exception:
        return content


def _resolve_model_path(path: str) -> tuple:
    """
    Resolve a Hugging Face cache path.
    - If config.json exists under path, return (path, None).
    - If path is a cache root (refs/main and blobs/), resolve snapshot path or cache_dir,
      and return (resolved_path, cache_dir).
    """
    path = os.path.abspath(path)
    config_path = os.path.join(path, "config.json")
    if os.path.isfile(config_path):
        return path, None

    refs_main = os.path.join(path, "refs", "main")
    snapshots_dir = os.path.join(path, "snapshots")
    if os.path.isfile(refs_main):
        with open(refs_main, "r", encoding="utf-8") as f:
            revision = f.read().strip()
        snapshot_path = os.path.join(snapshots_dir, revision)
        if os.path.isfile(os.path.join(snapshot_path, "config.json")):
            return snapshot_path, None
        # Cache root exists but snapshots/ is missing: use model id + cache_dir for resolution.
        parent_cache = os.path.dirname(path)
        return None, parent_cache
    return path, None


def _get_qwen_manager():
    """Lazy-load local Qwen model. Qwen3 requires transformers>=4.51."""
    global _qwen_manager, _qwen_tokenizer
    if _qwen_manager is not None:
        return _qwen_manager, _qwen_tokenizer
    _apply_hf_filelock_workaround()
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("Qwen backend requires: pip install torch transformers")
    # Current transformers requires PyTorch >= 2.4 for loading this model.
    torch_ver = getattr(torch, "__version__", "0")
    def _v(s):
        return [int(x) if x.isdigit() else 0 for x in (s or "0").split(".")[:3]]
    if _v(torch_ver) < [2, 4, 0]:
        raise ImportError(
            "Local Qwen requires PyTorch>=2.4, current version is %s. Upgrade with: pip install -U 'torch>=2.4'" % torch_ver
        )
    # Qwen3 architecture requires transformers >= 4.51.
    def _parse_version(v):
        parts = []
        for x in (v or "0").split(".")[:3]:
            parts.append(int(x) if x.isdigit() else 0)
        return (parts + [0, 0])[:3]
    tver = _parse_version(getattr(transformers, "__version__", "0"))
    if tver < [4, 51, 0]:
        raise ValueError(
            "Qwen3 requires transformers>=4.51, current version is {}. Upgrade with: pip install -U 'transformers>=4.51'".format(
                getattr(transformers, "__version__", "unknown")
            )
        )

    model_name = QWEN_MODEL
    resolved, cache_dir = _resolve_model_path(model_name)
    # If resolved is a local model directory (with config.json), use it directly.
    if resolved and os.path.isfile(os.path.join(resolved, "config.json")):
        load_path = resolved
        load_kw = {"trust_remote_code": True}
    elif cache_dir:
        load_path = "Qwen/Qwen3-4B-Instruct-2507"
        load_kw = {"trust_remote_code": True, "cache_dir": cache_dir}
    else:
        # For Hub id: snapshot_download first, then load locally to avoid path ambiguity and lock issues.
        cache_root = _cache_dir_with_local_locks()
        try:
            load_path = _hub_id_to_local_path(model_name, cache_root)
        except (PermissionError, OSError) as e:
            fallback = os.path.expanduser("~/.cache/huggingface/hub")
            print(f"[llm_backend] Cache directory unavailable, switching to {fallback}: {e}")
            load_path = _hub_id_to_local_path(model_name, fallback)
        load_kw = {"trust_remote_code": True}
    print(f"[llm_backend] Loading local Qwen: {load_path}" + (f" (cache_dir={load_kw.get('cache_dir')})" if load_kw.get("cache_dir") else ""))
    # If cache_dir is not writable, retry with user-local cache.
    fallback_cache = os.path.expanduser("~/.cache/huggingface/hub")
    try:
        _qwen_tokenizer = AutoTokenizer.from_pretrained(load_path, use_fast=False, **load_kw)
    except PermissionError as e:
        if load_kw.get("cache_dir") and load_kw["cache_dir"] != fallback_cache:
            print(f"[llm_backend] Cache permission error, switching to {fallback_cache}: {e}")
            load_kw = {**load_kw, "cache_dir": fallback_cache}
            _qwen_tokenizer = AutoTokenizer.from_pretrained(load_path, use_fast=False, **load_kw)
        else:
            raise
    dtype = getattr(torch, "bfloat16", torch.float16)
    # New transformers uses dtype; older versions use torch_dtype.
    # device_map requires accelerate; fallback is manual .to(device).
    def _load_model(kw):
        try:
            return AutoModelForCausalLM.from_pretrained(load_path, **kw)
        except ValueError as e:
            if "accelerate" not in str(e).lower():
                raise
            kw = dict(kw)
            kw.pop("device_map", None)
            kw.pop("max_memory", None)
            model = AutoModelForCausalLM.from_pretrained(load_path, **kw)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return model.to(device)
    load_model_kw = {**load_kw, "device_map": "auto", "dtype": dtype}
    if QWEN_MAX_GPU_MEMORY:
        # Limit per-GPU memory, e.g. {"0": "20GiB"}; requires accelerate.
        load_model_kw["max_memory"] = {0: QWEN_MAX_GPU_MEMORY}
    try:
        _qwen_manager = _load_model(load_model_kw)
    except TypeError:
        load_model_kw.pop("dtype", None)
        load_model_kw["torch_dtype"] = dtype
        _qwen_manager = _load_model(load_model_kw)
    _qwen_manager.eval()  # eval mode can reduce memory usage.
    print("[llm_backend] Qwen loaded")
    return _qwen_manager, _qwen_tokenizer


def _qwen_generate(messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    """Generate response with local Qwen. messages format is OpenAI-compatible."""
    model, tokenizer = _get_qwen_manager()
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    gen_kwargs = {
        "max_new_tokens": QWEN_MAX_NEW_TOKENS,
        "repetition_penalty": 1.05,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.8
    # inference_mode generally uses less memory than no_grad.
    torch = __import__("torch")
    with torch.inference_mode():
        generated = model.generate(**model_inputs, **gen_kwargs)
    output_ids = generated[0][model_inputs.input_ids.shape[1] :]
    output_id_list = output_ids.tolist() if hasattr(output_ids, "tolist") else list(output_ids)

    # Keep thinking mode enabled, but return only the final answer part after </think> when present.
    response = ""
    try:
        if QWEN_THINK_END_TOKEN_ID in output_id_list:
            end_idx = len(output_id_list) - 1 - output_id_list[::-1].index(QWEN_THINK_END_TOKEN_ID)
            answer_ids = output_id_list[end_idx + 1 :]
            response = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    except Exception:
        response = ""

    if not response:
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    # Remove possible chat-template suffixes.
    for suf in ("<|im_end|>", "<|endoftext|>"):
        if response.endswith(suf):
            response = response[: -len(suf)].strip()
    return response


def llm_complete(
    model: str,
    prompt: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.0,
) -> str:
    """
    Completion for local backends (qwen or vllm).
    Compatible with ChatGPT_API: messages = chat_history + [user, prompt].
    """
    messages = _build_messages(prompt=prompt, chat_history=chat_history)
    if LLM_BACKEND == "vllm":
        return _vllm_generate(messages, model=model, temperature=temperature)
    return _qwen_generate(messages, temperature=temperature)


async def llm_complete_async(
    model: str,
    prompt: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.0,
) -> str:
    """Async Qwen call implemented via thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: llm_complete(model, prompt, chat_history=chat_history, temperature=temperature),
    )


def llm_complete_with_finish_reason(
    model: str,
    prompt: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.0,
) -> tuple:
    """Return (content, finish_reason). Local qwen/vllm returns finish_reason='finished'."""
    content = llm_complete(model, prompt, chat_history=chat_history, temperature=temperature)
    return (content, "finished")


def _get_outlines_model():
    """
    Build/reuse an Outlines Transformers wrapper around the already-loaded local Qwen model.
    """
    global _outlines_model
    if _outlines_model is not None:
        return _outlines_model
    try:
        import outlines
    except ImportError:
        raise ImportError("Outlines support requires: pip install outlines")
    model, tokenizer = _get_qwen_manager()
    _outlines_model = outlines.models.Transformers(model, tokenizer)
    return _outlines_model


def _trace_structured_json(
    prompt: str,
    schema_key: str,
    result_obj: Any,
    backend_label: str,
) -> None:
    """
    Optional debug trace: save every structured-generation result to disk.
    Enable by setting QWEN_JSON_TRACE_DIR to a writable directory.
    """
    trace_dir = os.getenv("QWEN_JSON_TRACE_DIR", "").strip()
    if not trace_dir:
        return
    global _qwen_json_trace_seq
    _qwen_json_trace_seq += 1
    os.makedirs(trace_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_path = os.path.join(trace_dir, f"{ts}_{_qwen_json_trace_seq:06d}.json")
    payload = {
        "timestamp_utc": ts,
        "seq": _qwen_json_trace_seq,
        "backend": backend_label,
        "schema_key": schema_key,
        "prompt": prompt,
        "result": result_obj,
    }
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[llm_backend] Structured JSON trace saved: {out_path}")
    except Exception as e:
        print(f"[llm_backend] Failed to write structured JSON trace: {e}")


def llm_complete_json_schema(
    model: str,
    prompt: str,
    json_schema: Dict[str, Any],
    whitespace_pattern: str = r"[ ]?",
) -> Any:
    """
    Structured generation for local backends.
    - qwen backend: Outlines-constrained decoding.
    - vllm backend: guided_json via OpenAI-compatible vLLM API.
    Returns parsed Python object (dict/list) when possible.
    """
    if LLM_BACKEND == "vllm":
        messages = _build_messages(prompt=prompt, chat_history=None)
        result = _vllm_generate_json_schema(messages=messages, model=model, json_schema=json_schema)
        schema_key = json.dumps(json_schema, sort_keys=True, ensure_ascii=False)
        _trace_structured_json(prompt, schema_key, result, "vllm.guided_json")
        return result
    if LLM_BACKEND != "qwen":
        raise ValueError("llm_complete_json_schema is only available when LLM_BACKEND in {qwen, vllm}")
    outlines_model = _get_outlines_model()
    try:
        import outlines
    except ImportError:
        raise ImportError("Outlines support requires: pip install outlines")
    schema_key = json.dumps(json_schema, sort_keys=True, ensure_ascii=False)
    if schema_key not in _outlines_json_generators:
        # Outlines API compatibility:
        # - older versions: outlines.generate.json(model, schema, ...)
        # - newer versions: outlines.Generator(model, output_type=outlines.json_schema(schema))
        if hasattr(outlines, "generate") and hasattr(outlines.generate, "json"):
            _outlines_json_generators[schema_key] = outlines.generate.json(
                outlines_model,
                schema_key,
                whitespace_pattern=whitespace_pattern,
            )
        else:
            output_type = outlines.json_schema(schema_key)
            _outlines_json_generators[schema_key] = outlines.Generator(
                outlines_model,
                output_type=output_type,
            )
    generator = _outlines_json_generators[schema_key]
    result = generator(prompt, max_new_tokens=QWEN_MAX_NEW_TOKENS)
    backend_label = "outlines.generate.json" if hasattr(outlines, "generate") and hasattr(getattr(outlines, "generate"), "json") else "outlines.Generator"
    if hasattr(result, "model_dump"):
        out = result.model_dump()
        _trace_structured_json(prompt, schema_key, out, backend_label)
        return out
    if isinstance(result, str):
        try:
            out = json.loads(result)
            _trace_structured_json(prompt, schema_key, out, backend_label)
            return out
        except Exception:
            _trace_structured_json(prompt, schema_key, result, backend_label)
            return result
    _trace_structured_json(prompt, schema_key, result, backend_label)
    return result


def count_tokens_llm(text: str, model: Optional[str] = None) -> int:
    """
    Count tokens, consistent with utils.count_tokens behavior.
    - backend=openai: use tiktoken (model selects encoding).
    - backend=qwen: use Qwen tokenizer (model argument ignored).
    - backend=vllm: fallback to tiktoken approximation.
    """
    if not text:
        return 0
    if LLM_BACKEND == "qwen":
        _, tokenizer = _get_qwen_manager()
        return len(tokenizer.encode(text, add_special_tokens=False))
    import tiktoken
    m = model or "gpt-4o-2024-11-20"
    try:
        enc = tiktoken.encoding_for_model(m)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
