"""
Shared PaddleOCR engine for manga-image-translator (custom build).

PaddleOCR gives far better text DETECTION + recognition on product / document
images than the default DBConvNeXt detector (finds small note lines, reads
dimension labels, ignores decorative dashed lines).

Two compatibility shims are required for this container:
1. NumPy 2.0 removed ``np.sctypes`` which ``imgaug`` (a paddleocr dependency)
   still imports. We restore it before importing paddleocr.
2. ``paddlepaddle==2.6.2`` crashes with SIGILL inside the ``self_attention_fuse_pass``
   IR-optimization pass on some CPUs. We disable IR optimization by monkeypatching
   ``paddle.inference.Config.switch_ir_optim`` to always pass ``False``.

The engine is a lazily-initialised singleton (model load is expensive).
"""
import threading
import numpy as np

from .log import get_logger

logger = get_logger('paddle_engine')

_engine_lock = threading.Lock()
_engine = None


def _apply_numpy_shim():
    if not hasattr(np, "sctypes"):
        np.sctypes = {
            "float": [np.float16, np.float32, np.float64],
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "complex": [np.complex64, np.complex128],
            "others": [bool, object, bytes, str, np.void],
        }


def _disable_ir_optim():
    """Avoid the SIGILL in paddle 2.6.2's self_attention_fuse_pass on this CPU."""
    try:
        import paddle.inference as _pi
        if not getattr(_pi.Config, "_mit_ir_patched", False):
            _orig = _pi.Config.switch_ir_optim
            _pi.Config.switch_ir_optim = lambda self, x=True: _orig(self, False)
            _pi.Config._mit_ir_patched = True
    except Exception as e:
        logger.warning(f"Could not disable paddle IR optim: {e}")


class _PaddleEngine:
    def __init__(self, lang: str = "ch", use_gpu: bool = False):
        _apply_numpy_shim()
        _disable_ir_optim()
        import logging as _logging
        _logging.getLogger("ppocr").setLevel(_logging.ERROR)
        from paddleocr import PaddleOCR
        # use_angle_cls/cls disabled: product text is upright; avoids extra model.
        self._ocr = PaddleOCR(use_angle_cls=False, lang=lang, show_log=False,
                              use_gpu=use_gpu, enable_mkldnn=False)
        logger.info(f"PaddleOCR engine ready (lang={lang}, gpu={use_gpu})")

    def ocr(self, image: np.ndarray):
        """Run paddle detect+recognize. Returns list of (pts(4,2 int), text, conf)."""
        res = self._ocr.ocr(image, cls=False)
        out = []
        lines = res[0] if res and res[0] else []
        for box, rec in lines:
            try:
                text, conf = rec
            except Exception:
                continue
            pts = np.array(box, dtype=np.int64).reshape(-1, 2)
            out.append((pts, text, float(conf)))
        return out


def get_paddle_engine(lang: str = "ch", use_gpu: bool = False) -> "_PaddleEngine":
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = _PaddleEngine(lang=lang, use_gpu=use_gpu)
    return _engine
