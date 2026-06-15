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
import base64, io, os, pickle
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
    text_th = det_cfg.get("text_threshold")
    if text_th and float(text_th) <= 0.3:
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
    if engine == "paddle" or det_cfg.get("detector") == "paddle_ocr":
        det_cfg["detector"] = "paddle_ocr"
        rc.setdefault("ocr", {})["ocr"] = "paddle"

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
    try:
        async with httpx.AsyncClient(timeout=300) as c:
            r = await c.post(
                f"{SHARED}/simple_execute/translate",
                content=payload,
                headers={"Content-Type": "application/octet-stream"},
            )
    except Exception as e:
        raise HTTPException(503, detail=f"Shared API connection error: {e}")

    if r.status_code == 429:
        raise HTTPException(429, detail="Translator is busy, please retry")
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
        return {
            "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
            "regions": regions_meta,
            "region_count": len(regions_meta),
        }

    return Response(content=buf.getvalue(), media_type="image/png")


def _extract_regions_meta(result):
    """Extract per-region metadata from the MT Context result for diagnostics."""
    regions = getattr(result, "text_regions", None) or []
    out = []
    import numpy as _np

    def _box(region):
        for attr in ("xyxy", "min_rect", "unrotated_min_rect"):
            v = getattr(region, attr, None)
            if v is None:
                continue
            try:
                arr = _np.array(v).reshape(-1, 2)
                return {
                    "x_min": int(arr[:, 0].min()), "y_min": int(arr[:, 1].min()),
                    "x_max": int(arr[:, 0].max()), "y_max": int(arr[:, 1].max()),
                }
            except Exception:
                continue
        return None

    for i, region in enumerate(regions):
        try:
            texts = getattr(region, "texts", None)
            lines = len(texts) if texts is not None else None
            out.append({
                "id": i,
                "text": getattr(region, "text", None),
                "translation": getattr(region, "translation", None),
                "font_size": int(getattr(region, "font_size", 0) or 0),
                "alignment": getattr(region, "alignment", None),
                "horizontal": bool(getattr(region, "horizontal", False)),
                "lines": lines,
                "box": _box(region),
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
