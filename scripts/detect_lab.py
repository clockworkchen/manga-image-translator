"""Detection-only recall lab: which detector/threshold actually finds every line?

Rendering quality is moot for text that was never detected. This runs ONLY the
detector (no OCR, no translation, no inpaint) so a sweep costs seconds instead
of ~90s per pipeline pass, and reports:

  * how many textlines came back
  * whether a set of --expect boxes (lines we know are missed) got covered
  * an annotated PNG of every detection

ASCII-only output on purpose (runs through docker exec on a cp936 console).

  python scripts/detect_lab.py --path /app/render_lab/page.jpg \
      --detector ctd --size 2048 --text-th 0.5 --box-th 0.7 --unclip 2.3 \
      --expect 146,464,240,484 --expect 320,570,400,620
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/app")

LAB_DIR = Path(os.environ.get("RENDER_LAB_DIR", "/app/render_lab"))


def _aabb(pts):
    a = np.asarray(pts).reshape(-1, 2).astype(float)
    return [a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()]


def _covered(expect, boxes, min_iou_of_expect=0.35) -> bool:
    """True if some detection covers most of the expected box.

    Intersection is normalised by the EXPECTED area, not the union: a detector
    that merges the line into a bigger block still counts as finding it.
    """
    ex1, ey1, ex2, ey2 = expect
    ea = max(1.0, (ex2 - ex1) * (ey2 - ey1))
    for b in boxes:
        ix = max(0, min(ex2, b[2]) - max(ex1, b[0]))
        iy = max(0, min(ey2, b[3]) - max(ey1, b[1]))
        if ix * iy / ea >= min_iou_of_expect:
            return True
    return False


async def run(args) -> int:
    from manga_translator.detection import dispatch as dispatch_detection

    img = cv2.imread(args.path)
    if img is None:
        print(f"cannot read {args.path}")
        return 1
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"image {img.shape[1]}x{img.shape[0]}  detector={args.detector} "
          f"size={args.size} text_th={args.text_th} box_th={args.box_th} "
          f"unclip={args.unclip}")

    textlines, mask_raw, mask = await dispatch_detection(
        args.detector, img_rgb, args.size, args.text_th, args.box_th,
        args.unclip, args.invert, args.gamma, args.rotate, args.auto_rotate,
        "cpu" if args.cpu else "cuda", verbose=False)

    boxes = [_aabb(t.pts) for t in textlines]
    print(f"textlines={len(textlines)}")
    for i, (t, b) in enumerate(zip(textlines, boxes)):
        print(f"  {i:>3} ({int(b[0])},{int(b[1])}) "
              f"{int(b[2]-b[0])}x{int(b[3]-b[1])} prob={getattr(t, 'prob', 0):.2f}")

    if args.blocks:
        # Show how the VLM OCR would group these boxes into bubble-level crops.
        # Over-grouping is the failure that matters: two bubbles in one crop get
        # transcribed as one block and their lines end up merged.
        from manga_translator.ocr.model_vlm import ModelVlmOCR
        groups = ModelVlmOCR._group_boxes(list(textlines))
        print(f"blocks={len(groups)}")
        for i, g in enumerate(groups):
            members = [k for k, b in enumerate(boxes)
                       if b[0] >= g[0] - 1 and b[1] >= g[1] - 1
                       and b[2] <= g[2] + 1 and b[3] <= g[3] + 1]
            print(f"  blk {i:>3} ({int(g[0])},{int(g[1])}) "
                  f"{int(g[2]-g[0])}x{int(g[3]-g[1])} boxes={members}")

    if args.ocr:
        # Detection boxes are only half the story: OCR can return empty text or
        # get filtered as "not valuable", and such a box is dropped silently -
        # the line looks like a detection miss but was actually detected.
        from manga_translator.ocr import dispatch as dispatch_ocr
        from manga_translator.config import OcrConfig
        recognised = await dispatch_ocr(
            args.ocr, img_rgb, list(textlines), OcrConfig(ocr=args.ocr),
            "cpu" if args.cpu else "cuda", verbose=False)
        # Print the recognized lines as-is. Do NOT try to map them back onto the
        # detection boxes: an OCR may split a multi-line box into one box per
        # line, so the two lists are different shapes and any positional
        # pairing silently mislabels every row.
        rboxes = [_aabb(t.pts) for t in recognised]
        print(f"after ocr: {len(textlines)} boxes -> {len(recognised)} lines")
        for i, (t, b) in enumerate(sorted(zip(recognised, rboxes),
                                          key=lambda it: (it[1][1], it[1][0]))):
            print(f"  ocr {i:>3} ({int(b[0])},{int(b[1])}) "
                  f"{int(b[2]-b[0])}x{int(b[3]-b[1])} text={str(t.text)[:44]!r}")
        boxes = rboxes

    if args.expect:
        hits = 0
        for e in args.expect:
            ok = _covered(e, boxes)
            hits += ok
            print(f"  expect ({e[0]},{e[1]})-({e[2]},{e[3]}): "
                  f"{'FOUND' if ok else 'MISSED'}")
        print(f"recall {hits}/{len(args.expect)}")

    LAB_DIR.mkdir(parents=True, exist_ok=True)
    ann = img.copy()
    for b in boxes:
        cv2.rectangle(ann, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                      (0, 0, 255), 2)
    for e in (args.expect or []):
        cv2.rectangle(ann, (e[0], e[1]), (e[2], e[3]), (0, 200, 0), 2)
    tag = args.tag or f"{args.detector}_{args.size}_{args.text_th}_{args.box_th}_{args.unclip}"
    cv2.imwrite(str(LAB_DIR / f"det_{tag}.png"), ann)
    if mask is not None:
        cv2.imwrite(str(LAB_DIR / f"det_{tag}_mask.png"), mask)
    if mask_raw is not None:
        # mask_raw is what actually drives erasing when a detector returns no
        # refined mask (both `default` and `ctd` do exactly that), so it is the
        # one to inspect when original text survives inpainting.
        print(f"mask_raw shape={mask_raw.shape} dtype={mask_raw.dtype} "
              f"min={mask_raw.min()} max={mask_raw.max()}")
        cv2.imwrite(str(LAB_DIR / f"det_{tag}_maskraw.png"), mask_raw)
    print(f"wrote det_{tag}.png (red=detected, green=expected)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--detector", default="ctd")
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--text-th", type=float, default=0.5)
    ap.add_argument("--box-th", type=float, default=0.7)
    ap.add_argument("--unclip", type=float, default=2.3)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--gamma", action="store_true")
    ap.add_argument("--rotate", action="store_true")
    ap.add_argument("--auto-rotate", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--ocr", default="", help="also run this OCR and show drops")
    ap.add_argument("--blocks", action="store_true",
                    help="show the VLM OCR bubble grouping of the boxes")
    ap.add_argument("--tag", default="")
    ap.add_argument("--expect", action="append", default=[],
                    help="x1,y1,x2,y2 of a line that must be detected")
    args = ap.parse_args()
    args.expect = [[int(v) for v in e.split(",")] for e in args.expect]
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
