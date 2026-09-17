"""
JSON REST proxy for manga-image-translator shared (pickle) API.
Part of the custom manga-image-translator build.

Exposes:  GET  /health        → {"ok": true}
          POST /queue-size    → int (0 = idle, 1 = busy)
          POST /translate/image → PNG image bytes
          POST /exec          → run shell command (debug)

Internal manga-translator shared API runs on port 5005.
This proxy listens on port 5003 (configurable via MT_PROXY_PORT env var).
"""
import asyncio, base64, io, os, pickle, time
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from PIL import Image

try:
    from manga_translator.config import Config as MTConfig
    _HAS_MT = True
except ImportError:
    _HAS_MT = False

app = FastAPI()
SHARED = os.environ.get("MT_SHARED_URL", "http://127.0.0.1:5005")
PROXY_PORT = int(os.environ.get("MT_PROXY_PORT", "5003"))
# A large image plus a slow LLM can legitimately take longer than five minutes.
# Disable the internal read timeout by default: the public API is asynchronous and
# callers poll the task record, while the shared worker already serializes work.
SHARED_READ_TIMEOUT_SEC = float(os.environ.get("MT_SHARED_READ_TIMEOUT_SEC", "0"))
SHARED_BUSY_WAIT_SEC = float(os.environ.get("MT_SHARED_BUSY_WAIT_SEC", "180"))
SHARED_BUSY_POLL_SEC = max(0.2, float(os.environ.get("MT_SHARED_BUSY_POLL_SEC", "1")))


def _shared_timeout():
    import httpx
    read_timeout = None if SHARED_READ_TIMEOUT_SEC <= 0 else SHARED_READ_TIMEOUT_SEC
    return httpx.Timeout(connect=10.0, read=read_timeout, write=60.0, pool=10.0)


def _connection_error_detail(exc: BaseException, elapsed_sec: float) -> str:
    message = str(exc).strip() or repr(exc)
    return (
        f"Shared API connection error ({type(exc).__name__}) after "
        f"{elapsed_sec:.1f}s: {message}"
    )


@app.get("/health")
async def health():
    return {"ok": True, "proxy": True}


@app.post("/queue-size")
async def queue_size():
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{SHARED}/is_locked")
            return 1 if r.json().get("locked") else 0
    except Exception:
        raise HTTPException(503, detail="Shared API not reachable")


@app.post("/translate/image")
async def translate_image(request: Request):
    import httpx
    req = await request.json()

    # ── Decode image ──────────────────────────────────────────────
    img_data = req.get("image", "")
    if img_data.startswith("data:"):
        img_data = img_data.split(",", 1)[1]
    try:
        raw = base64.b64decode(img_data)
        pil_image = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception as e:
        raise HTTPException(400, detail=f"Image decode error: {e}")

    # ── Build Config ──────────────────────────────────────────────
    if not _HAS_MT:
        raise HTTPException(500, detail="manga_translator not installed in container")
    config = MTConfig()
    rc = req.get("config", {}) or {}
    tr_cfg = rc.get("translator", {}) if rc else {}
    config.translator.translator = tr_cfg.get("translator") or req.get("translator") or "sugoi"
    config.translator.target_lang = tr_cfg.get("target_lang") or req.get("target_lang") or "CHS"
    llm_model = tr_cfg.get("llm_model") or tr_cfg.get("model") or req.get("llm_model")
    if llm_model:
        try:
            config.translator.llm_model = str(llm_model).strip()
        except Exception:
            pass
    llm_api_base = tr_cfg.get("llm_api_base")
    llm_api_key = tr_cfg.get("llm_api_key")
    for attr, val in (("llm_api_base", llm_api_base), ("llm_api_key", llm_api_key)):
        if val:
            try:
                setattr(config.translator, attr, str(val).strip())
            except Exception:
                pass
    # ── Detector config (text region detection) ───────────────────
    det_cfg = rc.get("detector", {}) or {}
    def _set(obj, attr, val):
        """Set attribute if it exists on the config object and val is truthy."""
        if val and hasattr(obj, attr):
            try: setattr(obj, attr, val)
            except Exception: pass

    # Smart defaults for small-text / product-image mode:
    # When text_threshold <= 0.3, lower box_threshold and widen detection boxes.
    # Verified on product images (testglass1): box_threshold=0.25 + unclip_ratio=3.5
    # captures small note lines AND avoids truncating long spec lines (e.g.
    # "适用场景：驾驶|出行|通勤" was being cut to "驾驶|出" with the old 2.5 ratio).
    requested_engine = (rc.get("engine") or req.get("engine") or "").lower()
    manga_requested = (requested_engine in ("manga", "vlm")
                       or det_cfg.get("detector") == "ensemble"
                       or (rc.get("ocr", {}) or {}).get("ocr") == "vlm")
    text_th = det_cfg.get("text_threshold")
    if not manga_requested and text_th and float(text_th) <= 0.3:
        if not det_cfg.get("box_threshold"):
            det_cfg["box_threshold"] = 0.25
        if not det_cfg.get("unclip_ratio"):
            det_cfg["unclip_ratio"] = 3.5
        # Product-image mode: enable smart overflow handling and disable font border
        render_cfg = rc.setdefault("render", {})
        if not render_cfg.get("overflow_strategy"):
            render_cfg["overflow_strategy"] = "cascade"
        if "disable_font_border" not in render_cfg:
            render_cfg["disable_font_border"] = True

    # Convenience: a top-level "engine":"paddle" (or detector=="paddle_ocr")
    # selects the PaddleOCR detector. PaddleOCR detects + recognizes in one pass,
    # so it MUST be paired with the "paddle" passthrough OCR (otherwise another
    # OCR would discard paddle's recognized text). PaddleOCR is much better at
    # product/document images (finds small note lines, ignores decorative dashes).
    engine = (rc.get("engine") or req.get("engine") or "").lower()
    # PaddleOCR is the DEFAULT detection+recognition engine for this build: it is
    # far better on product/document images and keeps a single engine for the
    # whole lifecycle. It is only skipped if the caller explicitly asks for a
    # different detector (e.g. detector="ctd"/"default" for manga).
    if not engine and not det_cfg.get("detector"):
        engine = "paddle"

    # engine="manga" (or detector=="ensemble" / ocr=="vlm"): the comic pipeline.
    # Paddle is tuned for product images and is the wrong tool on comic pages;
    # this pairs the ensemble detector (no single detector finds every balloon)
    # with the VLM recognizer (the local models mangle stylized lettering and
    # silently drop whole multi-line balloons). Thresholds are lowered because at
    # the stock 0.5/0.7 a real balloon on the reference page was never detected.
    if engine in ("manga", "vlm") or det_cfg.get("detector") == "ensemble" \
            or (rc.get("ocr", {}) or {}).get("ocr") == "vlm":
        if not det_cfg.get("detector"):
            det_cfg["detector"] = "ensemble"
        rc.setdefault("ocr", {})["ocr"] = "vlm"
        det_cfg.setdefault("detection_size", 2048)
        det_cfg.setdefault("text_threshold", 0.3)
        det_cfg.setdefault("box_threshold", 0.4)
        det_cfg.setdefault("unclip_ratio", 2.8)
        _rc_render = rc.setdefault("render", {})
        if "disable_font_border" not in _rc_render:
            _rc_render["disable_font_border"] = True
        if not _rc_render.get("overflow_strategy"):
            # "bubble", not "cascade": a balloon has no free space around it, so
            # the cascade rules (expand into the neighbour gap) let the text run
            # out through the outline. "bubble" keeps it inside the footprint the
            # original text occupied and shrinks to fit.
            _rc_render["overflow_strategy"] = "bubble"
    elif engine == "paddle" or det_cfg.get("detector") == "paddle_ocr":
        det_cfg["detector"] = "paddle_ocr"
        rc.setdefault("ocr", {})["ocr"] = "paddle"
        # Paddle = product-image mode: borderless text + cascade layout by default
        # (disable_font_border avoids the bold/outline look on re-rendered text).
        _rc_render = rc.setdefault("render", {})
        if "disable_font_border" not in _rc_render:
            _rc_render["disable_font_border"] = True
        if not _rc_render.get("overflow_strategy"):
            _rc_render["overflow_strategy"] = "cascade"

    _set(config.detector, "detector", det_cfg.get("detector"))
    _set(config.detector, "detection_size", det_cfg.get("detection_size"))
    _set(config.detector, "text_threshold", det_cfg.get("text_threshold"))
    _set(config.detector, "box_threshold", det_cfg.get("box_threshold"))
    _set(config.detector, "unclip_ratio", det_cfg.get("unclip_ratio"))

    # ── OCR config ────────────────────────────────────────────────
    ocr_cfg = rc.get("ocr", {}) or {}
    # If paddle detector is selected, force the paddle passthrough OCR.
    if config.detector.detector == "paddle_ocr":
        ocr_cfg["ocr"] = "paddle"
    _set(config.ocr, "ocr", ocr_cfg.get("ocr"))
    if ocr_cfg.get("prob") is not None:
        _set(config.ocr, "prob", ocr_cfg.get("prob"))
    for attr, key in (
        ("vlm_api_base", "vlm_api_base"), ("vlm_api_key", "vlm_api_key"),
        ("vlm_model", "vlm_model"), ("vlm_concurrency", "vlm_concurrency"),
        ("vlm_timeout", "vlm_timeout"),
    ):
        value = ocr_cfg.get(key)
        if value is not None and value != "":
            try:
                setattr(config.ocr, attr, value)
            except Exception:
                pass

    # ── Render overrides ──────────────────────────────────────────
    if rc.get("inpainter", {}).get("inpainter"):
        _set(config.inpainter, "inpainter", rc["inpainter"]["inpainter"])
    render_cfg = rc.get("render", {}) or {}
    for rkey in ("direction", "alignment", "disable_font_border", "font_size_offset",
                 "font_size_minimum", "no_hyphenation", "uppercase", "lowercase",
                 "overflow_strategy", "max_font_shrink_ratio"):
        _set(config.render, rkey, render_cfg.get(rkey))

    # ── chatgpt_config: override for local LLMs ──────────────────
    # simple_prompt=1 replaces complex doujin-translator role + Japanese few-shot
    # examples with a minimal direct instruction for product/document images.
    if rc.get("simple_prompt"):
        SIMPLE_TPL = (
            "You are a professional translation engine. "
            "Translate the following text into {to_lang}. "
            "Output ONLY the translated lines with the exact same <|N|> prefix numbers. "
            "Preserve the visual order of bracketed labels: if a source line starts with "
            "【...】, the translation must also start with its translated 【...】 group. "
            "Do NOT output the original. Do NOT explain. Just translate.\n"
            "Translate to {to_lang}:\n"
        )
        try:
            from omegaconf import OmegaConf
            chatgpt_cfg = OmegaConf.create({
                "ollama": {
                    "chat_system_template": SIMPLE_TPL,
                    "chat_sample": {},
                    "include_template": False,
                }
            })
            # chatgpt_config is a read-only @property; set _gpt_config directly
            if hasattr(config, "translator"):
                config.translator._gpt_config = chatgpt_cfg
        except Exception as e:
            import logging as _logging
            _logging.getLogger("json_proxy").warning(f"simple_prompt setup failed: {e}")

    # ── Call shared API via pickle ────────────────────────────────
    payload = pickle.dumps({"image": pil_image, "config": config})
    started_at = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=_shared_timeout()) as c:
            deadline = time.monotonic() + SHARED_BUSY_WAIT_SEC
            while True:
                r = await c.post(
                    f"{SHARED}/simple_execute/translate",
                    content=payload,
                    headers={"Content-Type": "application/octet-stream"},
                )
                if r.status_code != 429 or time.monotonic() >= deadline:
                    break
                # Shared rejects before consuming the pickle payload while locked,
                # so resubmitting the same bytes is safe. Never retry any other
                # status (especially safety, auth, quota or translation errors).
                await asyncio.sleep(SHARED_BUSY_POLL_SEC)
    except Exception as e:
        detail = _connection_error_detail(e, time.monotonic() - started_at)
        raise HTTPException(503, detail=detail) from e

    if r.status_code == 429:
        raise HTTPException(429, detail=f"Translator remained busy for {SHARED_BUSY_WAIT_SEC:.0f}s")
    if r.status_code != 200:
        raise HTTPException(r.status_code, detail=r.text[:2000])

    # ── Unpack result ─────────────────────────────────────────────
    try:
        result = pickle.loads(r.content)
        img_out = result.result if hasattr(result, "result") else result
        if img_out.mode == "RGBA":
            img_out = img_out.convert("RGB")
        buf = io.BytesIO()
        img_out.save(buf, format="PNG")
    except Exception as e:
        raise HTTPException(500, detail=f"Result decode error: {e}")

    # ── Debug mode: return JSON with real text-region metadata ─────
    # This is the ground truth for the diagnostic (translate_debug.py):
    # the actual detected regions, their translations and the font sizes
    # MT used for rendering — far more reliable than pixel analysis.
    if req.get("debug"):
        regions_meta = _extract_regions_meta(result)
        raw_dets = _extract_raw_detections(result)
        return {
            "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
            "regions": regions_meta,
            "region_count": len(regions_meta),
            "raw_detections": raw_dets,
            "raw_detection_count": len(raw_dets),
            "mask": _encode_debug_image(getattr(result, "mask", None), mode="L"),
            "inpainted": _encode_debug_image(getattr(result, "img_inpainted", None)),
        }

    return Response(content=buf.getvalue(), media_type="image/png")


def _encode_debug_image(value, mode="RGB"):
    if value is None:
        return None
    try:
        import numpy as _np
        image = Image.fromarray(_np.asarray(value).astype(_np.uint8), mode=mode)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
    except Exception:
        return None


def _extract_raw_detections(result):
    """Extract the RAW detector textlines (pre-merge) for diagnostics.

    PaddleOCR detects every line separately; MT then merges/filters them (e.g.
    English/numeric lines are skipped for CHS->ENG). Exposing the raw detections
    lets the diagnostic show the TRUE detection coverage vs the final regions.
    """
    import numpy as _np
    out = []
    textlines = getattr(result, "textlines", None) or []
    for i, t in enumerate(textlines):
        try:
            pts = getattr(t, "pts", None)
            arr = _np.array(pts).reshape(-1, 2)
            out.append({
                "id": i,
                "text": getattr(t, "text", None),
                "x_min": int(arr[:, 0].min()), "y_min": int(arr[:, 1].min()),
                "x_max": int(arr[:, 0].max()), "y_max": int(arr[:, 1].max()),
            })
        except Exception:
            continue
    return out


def _extract_regions_meta(result):
    """Extract per-region metadata from the MT Context result for diagnostics."""
    regions = getattr(result, "text_regions", None) or []
    out = []
    import numpy as _np

    def _aabb(v):
        if v is None:
            return None
        try:
            arr = _np.array(v).reshape(-1, 2)
            return {
                "x_min": int(arr[:, 0].min()), "y_min": int(arr[:, 1].min()),
                "x_max": int(arr[:, 0].max()), "y_max": int(arr[:, 1].max()),
            }
        except Exception:
            return None

    def _box(region):
        for attr in ("xyxy", "min_rect", "unrotated_min_rect"):
            b = _aabb(getattr(region, attr, None))
            if b is not None:
                return b
        return None

    for i, region in enumerate(regions):
        try:
            source_lines = getattr(region, "lines", None)
            lines = len(source_lines) if source_lines is not None else None
            # Final rendered box (post expand/normalize/overlap-resolve) — the
            # box the translated text is actually warped into. This is the real
            # ground truth for overlap and displayed-size checks.
            render_box = _aabb(getattr(region, "_render_dst", None))
            out.append({
                "id": i,
                "text": getattr(region, "text", None),
                "translation": getattr(region, "translation", None),
                "font_size": int(getattr(region, "font_size", 0) or 0),
                "orig_font_size": int(getattr(region, "_orig_font_size", 0) or 0),
                "alignment": getattr(region, "alignment", None),
                "fg_color": [int(v) for v in region.get_font_colors()[0]],
                "bg_color": [int(v) for v in region.get_font_colors()[1]],
                "horizontal": bool(getattr(region, "horizontal", False)),
                "angle": float(getattr(region, "angle", 0) or 0),
                "lines": lines,
                "render_lines": int(getattr(region, "_render_lines", 0) or 0) or None,
                "box": _box(region),
                "render_box": render_box,
                "bubble_group": getattr(region, "_bubble_group", None),
                "bubble_original_lines": getattr(region, "_bubble_original_lines", None),
                "bubble_retained_lines": getattr(region, "_bubble_retained_lines", None),
                "bubble_visible_ink_height": getattr(region, "_bubble_visible_ink_height", None),
            })
        except Exception as e:
            out.append({"id": i, "error": str(e)})
    return out


@app.post("/exec")
async def exec_command(request: Request):
    """Debug endpoint: run arbitrary shell command inside the container."""
    import subprocess, shlex
    body = await request.json()
    cmd = body.get("cmd", "")
    timeout = min(int(body.get("timeout", 30)), 120)
    if not cmd:
        return {"exit_code": 1, "stdout": "", "stderr": "No command provided"}
    try:
        proc = subprocess.run(
            ["/bin/bash", "-c", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"exit_code": -1, "stdout": "", "stderr": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
