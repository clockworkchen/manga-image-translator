"""Offline render lab: iterate on layout/rendering without re-running inference.

A full pipeline pass (detect + OCR + translate + inpaint) costs ~90s, which makes
tuning the renderer painfully slow. This script splits that in two:

  fixture  run the pipeline ONCE through the shared API with renderer=none and
           pickle the resulting Context pieces (text_regions, img_inpainted,
           img_rgb, raw textlines) to /app/render_lab/<name>.pkl
  render   load the fixture and call rendering.dispatch() directly, then report
           per-region metrics and dump annotated PNGs
  probe    dump what _measure_ink_height() sees per region (row runs, Otsu mask)
  merge    replay textline_merge with different tolerances on the raw textlines

All text is ASCII on purpose: this runs via `docker exec` from a cp936 Windows
console, and non-ASCII output/soure has silently corrupted this file before.

Run inside the manga-translator container:
  python scripts/render_lab.py fixture --url <image-url> --name repro
  python scripts/render_lab.py render --name repro
  python scripts/render_lab.py render --name repro --strategy wrap
"""
from __future__ import annotations

import argparse
import asyncio
import io
import os
import pickle
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/app")

LAB_DIR = Path(os.environ.get("RENDER_LAB_DIR", "/app/render_lab"))
SHARED = os.environ.get("MT_SHARED_URL", "http://127.0.0.1:5005")


# --------------------------------------------------------------------------- #
# fixture
# --------------------------------------------------------------------------- #
def cmd_fixture(args) -> int:
    import httpx
    from PIL import Image
    from manga_translator.config import Config

    if args.path:
        img_bytes = Path(args.path).read_bytes()
    else:
        with urllib.request.urlopen(args.url, timeout=120) as r:
            img_bytes = r.read()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    print(f"image {pil.size} {len(img_bytes)} bytes")

    cfg = {
        # Thresholds matter as much as the detector choice: at the stock
        # 0.5/0.7/2.3 the "GO AHEAD~" balloon is not detected at all, its box
        # never reaches OCR, and the line is left in English on the page.
        "detector": {"detector": args.detector, "detection_size": 2048,
                     "text_threshold": args.text_th,
                     "box_threshold": args.box_th,
                     "unclip_ratio": args.unclip},
        "ocr": {"ocr": args.ocr},
        "translator": {"translator": "custom_openai",
                       "target_lang": args.target_lang},
        # renderer=none stops after inpainting. Otherwise the pipeline renders
        # into img_inpainted itself and the fixture's "erased" image already
        # carries text; the lab would then draw on top of it and every ink
        # measurement would be the diff of two renders instead of one.
        "render": {"renderer": "none",
                   "disable_font_border": True,
                   "overflow_strategy": args.strategy,
                   "alignment": "center"},
    }
    config = Config(**cfg)
    payload = pickle.dumps({"image": pil, "config": config})
    print(f"POST {SHARED}/simple_execute/translate (this takes ~90s) ...")
    with httpx.Client(timeout=httpx.Timeout(connect=10, read=None, write=60, pool=10)) as c:
        resp = c.post(f"{SHARED}/simple_execute/translate", content=payload,
                      headers={"Content-Type": "application/octet-stream"})
    if resp.status_code != 200:
        print(f"FAIL HTTP {resp.status_code}: {resp.text[:500]}")
        return 1
    ctx = pickle.loads(resp.content)
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    out = LAB_DIR / f"{args.name}.pkl"
    fixture = {
        "img_inpainted": np.asarray(ctx.img_inpainted),
        "img_rgb": np.asarray(ctx.img_rgb),
        "text_regions": ctx.text_regions,
        # raw detections, so `merge` can replay textline_merge offline
        "textlines": getattr(ctx, "textlines", None),
        # The mask is what inpainting erased. Without it, "the balloon turned
        # into artwork" is indistinguishable from a rendering bug, and the two
        # need opposite fixes.
        "mask": None if getattr(ctx, "mask", None) is None else np.asarray(ctx.mask),
        "mask_raw": None if getattr(ctx, "mask_raw", None) is None else np.asarray(ctx.mask_raw),
    }
    with out.open("wb") as fh:
        pickle.dump(fixture, fh)
    cv2.imwrite(str(LAB_DIR / f"{args.name}_inpainted.png"),
                cv2.cvtColor(fixture["img_inpainted"], cv2.COLOR_RGB2BGR))
    for key in ("mask", "mask_raw"):
        if fixture[key] is not None:
            cv2.imwrite(str(LAB_DIR / f"{args.name}_{key}.png"), fixture[key])
            print(f"{key}: shape={fixture[key].shape} "
                  f"coverage={(fixture[key] > 0).mean():.4f}")
    print(f"fixture saved {out}  regions={len(ctx.text_regions)}")
    for i, r in enumerate(ctx.text_regions):
        print(f"  #{i:2d} lines={_nlines(r)} font={r.font_size} "
              f"text={r.text[:30]!r} -> {str(r.translation)[:30]!r}")
    return 0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _nlines(region) -> int:
    lines = getattr(region, "lines", None)
    return len(lines) if lines is not None else 1


def ink_stats(before: np.ndarray, after: np.ndarray, box) -> dict:
    """Measure what rendering actually painted: line count and median line height.

    The renderer warps its canvas onto dst_points, so region.font_size says
    nothing about the size finally seen on the page. Only the pixels do.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = before.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return {"lines": 0, "line_h": 0, "bbox": None}
    d = cv2.absdiff(before[y1:y2, x1:x2], after[y1:y2, x1:x2])
    if d.ndim == 3:
        d = d.max(axis=2)
    changed = d > 24
    if not changed.any():
        return {"lines": 0, "line_h": 0, "bbox": None}
    rows = np.where(changed.sum(axis=1) > 1)[0]
    cols = np.where(changed.sum(axis=0) > 1)[0]
    bbox = [x1 + int(cols[0]), y1 + int(rows[0]),
            x1 + int(cols[-1]) + 1, y1 + int(rows[-1]) + 1]
    row_ink = changed.sum(axis=1) > max(1, changed.shape[1] * 0.01)
    runs, run = [], 0
    for v in row_ink:
        if v:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    tallest = max(runs) if runs else 0
    real = [r for r in runs if r >= max(4, tallest * 0.4)] or runs
    real.sort()
    return {"lines": len(real),
            "line_h": (real[len(real) // 2] if real else 0),
            "bbox": bbox}


def aabb(pts):
    a = np.asarray(pts).reshape(-1, 2).astype(float)
    return [a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()]


def _fmt(b):
    if b is None:
        return "-"
    return f"{int(b[2]-b[0])}x{int(b[3]-b[1])}"


def _orig_line_h(img_rgb, region, box):
    # Use the renderer's own basis, not a second implementation: measuring
    # src_h differently here made the ratio column lie (it reported 56px for a
    # three-line bubble whose lines are 15px, so a correct render looked 0.34x).
    from manga_translator.rendering import _per_line_height
    return _per_line_height(region, box, img_rgb)


def _pick_font() -> str:
    from manga_translator.utils import BASE_PATH
    for cand in ("fonts/anime_ace_3.ttf", "fonts/msyh.ttc",
                 "fonts/NotoSansMonoCJK-VF.ttf.ttc"):
        p = os.path.join(BASE_PATH, cand)
        if os.path.isfile(p):
            return p
    return os.path.join(BASE_PATH, "fonts/anime_ace_3.ttf")


def _render(inpainted, regions, img_rgb, font, strategy):
    from manga_translator.rendering import dispatch
    return asyncio.run(dispatch(
        inpainted.copy(), regions, font_path=font,
        font_size_fixed=None, font_size_offset=0, font_size_minimum=-1,
        hyphenate=True, render_mask=None, line_spacing=None,
        disable_font_border=True, overflow_strategy=strategy,
        max_font_shrink_ratio=0.5, original_img=img_rgb,
    ))


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def cmd_render(args) -> int:
    from manga_translator.rendering import _estimate_bubble_bounds

    with (LAB_DIR / f"{args.name}.pkl").open("rb") as fh:
        fx = pickle.load(fh)
    regions = fx["text_regions"]
    inpainted = fx["img_inpainted"]
    img_rgb = fx["img_rgb"]

    if args.translations:
        # override translations from a "<index>: <text>" file, for A/B runs
        for line in Path(args.translations).read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            idx, txt = line.split(":", 1)
            regions[int(idx)].translation = txt.strip()

    if args.only >= 0:
        regions = [regions[args.only]]

    font = _pick_font()
    out = _render(inpainted, regions, img_rgb, font, args.strategy)

    LAB_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.name}_{args.strategy}"
    cv2.imwrite(str(LAB_DIR / f"{tag}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    print(f"\nstrategy={args.strategy}  regions={len(regions)}")
    print(f"{'#':>3} {'det_box':>11} {'bubble':>11} {'src_ln':>6} {'out_ln':>6} "
          f"{'src_h':>6} {'out_h':>6} {'ratio':>6} {'escape':>8}  translation")
    bad_overflow = bad_lines = bad_size = 0
    annotated = cv2.cvtColor(out, cv2.COLOR_RGB2BGR).copy()
    for i, r in enumerate(regions):
        det = aabb(r.min_rect)
        bub = _estimate_bubble_bounds(img_rgb, det)
        # Render this region ALONE: measuring on the full-page render mixes in
        # ink from neighbours whose boxes overlap this one.
        solo = _render(inpainted, [r], img_rgb, font, args.strategy)
        st = ink_stats(inpainted, solo, [0, 0, solo.shape[1], solo.shape[0]])
        src_lines = _nlines(r)
        src_h = _orig_line_h(img_rgb, r, det)
        ratio = (st["line_h"] / src_h) if src_h else 0
        esc = ""
        if st["bbox"] and bub:
            ex = max(0, bub[0] - st["bbox"][0]) + max(0, st["bbox"][2] - bub[2])
            ey = max(0, bub[1] - st["bbox"][1]) + max(0, st["bbox"][3] - bub[3])
            if ex > 2 or ey > 2:
                esc = f"x{int(ex)}y{int(ey)}"
                bad_overflow += 1
        if src_lines >= 2 and st["lines"] and st["lines"] < src_lines:
            bad_lines += 1
        if ratio and not (0.75 <= ratio <= 1.3):
            bad_size += 1
        print(f"{i:>3} {_fmt(det):>11} {_fmt(bub):>11} {src_lines:>6} {st['lines']:>6} "
              f"{src_h:>6.1f} {st['line_h']:>6} {ratio:>6.2f} {esc:>8}  "
              f"{str(r.translation)[:24]!r}")
        if bub:
            cv2.rectangle(annotated, (int(bub[0]), int(bub[1])),
                          (int(bub[2]), int(bub[3])), (0, 200, 0), 2)
        cv2.rectangle(annotated, (int(det[0]), int(det[1])),
                      (int(det[2]), int(det[3])), (0, 0, 255), 1)
    cv2.imwrite(str(LAB_DIR / f"{tag}_annotated.png"), annotated)
    print(f"\noverflow {bad_overflow}/{len(regions)}   "
          f"collapsed_lines {bad_lines}/{len(regions)}   "
          f"size_off(>25%) {bad_size}/{len(regions)}")
    print(f"wrote {LAB_DIR / (tag + '.png')} and {tag}_annotated.png "
          f"(green=bubble, red=detection box)")
    return 0


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def cmd_merge(args) -> int:
    """Replay textline_merge offline with different tolerances.

    Under-merging is what shifts every translation by one: each stray fragment
    becomes its own query, the LLM answers fewer lines than it was asked, and
    zip(regions, translations) silently misaligns from there on.
    """
    from manga_translator.textline_merge import merge_bboxes_text_region
    from manga_translator.utils.generic import quadrilateral_can_merge_region

    with (LAB_DIR / f"{args.name}.pkl").open("rb") as fh:
        fx = pickle.load(fh)
    tls = fx.get("textlines")
    if not tls:
        print("fixture has no 'textlines'; regenerate the fixture")
        return 1
    img = fx["img_rgb"]
    h, w = img.shape[:2]
    print(f"raw textlines={len(tls)}")
    for i, t in enumerate(tls):
        bb = t.aabb
        print(f"  {i:>3} ({bb.x},{bb.y}) {bb.w}x{bb.h} fs={t.font_size} "
              f"dir={t.direction} text={str(getattr(t, 'text', ''))[:34]!r}")

    def gate(a, b, fs_tol, ar_tol, gap1, gap2):
        """Return the reason the pair is rejected, or None if it can merge."""
        from shapely.geometry import Polygon as _P
        cs = min(a.font_size, b.font_size)
        dist = _P(a.pts).distance(_P(b.pts))
        if dist > 2 * cs:
            return f"distance {dist:.0f} > 2*{cs}"
        if max(a.font_size, b.font_size) / cs > fs_tol:
            return f"font_size_ratio {max(a.font_size,b.font_size)/cs:.2f} > {fs_tol}"
        b1, b2 = a.aabb, b.aabb
        both_h = (b1.w > b1.h * 1.2 and b2.w > b2.h * 1.2)
        if both_h:
            xg = abs((b1.x + b1.w / 2) - (b2.x + b2.w / 2))
            yg = abs((b1.y + b1.h / 2) - (b2.y + b2.h / 2))
            if xg > cs * 6 and xg > yg * 3:
                return f"cross-column x_gap={xg:.0f} y_gap={yg:.0f}"
        if a.aspect_ratio > ar_tol and b.aspect_ratio < 1.0 / ar_tol:
            return "aspect_ratio a/b"
        if b.aspect_ratio > ar_tol and a.aspect_ratio < 1.0 / ar_tol:
            return "aspect_ratio b/a"
        if dist >= cs * gap1:
            return f"char_gap dist {dist:.0f} >= {cs}*{gap1}"
        return None

    print(f"\npairwise gate (fs_tol={args.fs_tol} ar_tol={args.ar_tol} "
          f"gap1={args.gap1} gap2={args.gap2}):")
    from shapely.geometry import Polygon as _P
    for i in range(len(tls)):
        for j in range(i + 1, len(tls)):
            a, b = tls[i], tls[j]
            if _P(a.pts).distance(_P(b.pts)) > 3 * min(a.font_size, b.font_size):
                continue  # far apart, not interesting
            why = gate(a, b, args.fs_tol, args.ar_tol, args.gap1, args.gap2)
            ok = quadrilateral_can_merge_region(
                a, b, aspect_ratio_tol=args.ar_tol, font_size_ratio_tol=args.fs_tol,
                char_gap_tolerance=args.gap1, char_gap_tolerance2=args.gap2)
            mark = "MERGE" if ok else "  -  "
            print(f"  {i:>3}+{j:<3} {mark}  {str(getattr(a,'text',''))[:20]!r} | "
                  f"{str(getattr(b,'text',''))[:20]!r}   {why or ''}")

    regions = list(merge_bboxes_text_region(
        list(tls), w, h,
        font_size_ratio_tol=args.fs_tol, aspect_ratio_tol=args.ar_tol,
        char_gap_tolerance=args.gap1, char_gap_tolerance2=args.gap2,
        stacked_font_ratio_slack=args.slack))
    print(f"\nmerged into {len(regions)} regions:")
    for i, (txtlns, _, _) in enumerate(regions):
        txt = " / ".join(str(t.text) for t in txtlns)
        print(f"  #{i:2d} lines={len(txtlns)}  {txt[:80]!r}")
    return 0


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #
def cmd_probe(args) -> int:
    """Show exactly what _measure_ink_height() sees for every region.

    The cascade layout derives per-line height from the ORIGINAL ink, so a bad
    measurement here is what makes rendered text 2x too big or too small.
    """
    from manga_translator.rendering import _measure_ink_height

    with (LAB_DIR / f"{args.name}.pkl").open("rb") as fh:
        fx = pickle.load(fh)
    img = fx["img_rgb"]
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(fx["text_regions"]):
        box = aabb(r.min_rect)
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = img[y1:y2, x1:x2]
        g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverted = th.mean() > 127
        if inverted:
            th = 255 - th
        row_has_ink = (th > 0).sum(axis=1) > max(2, th.shape[1] * 0.02)
        runs, run = [], 0
        for v in row_has_ink:
            if v:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        if run:
            runs.append(run)
        print(f"#{i:2d} box=({x1},{y1}) {x2-x1}x{y2-y1} lines={_nlines(r)} "
              f"otsu_inverted={inverted} ink_rows={int(row_has_ink.sum())} "
              f"runs={runs} -> _measure_ink_height={_measure_ink_height(img, box)}")
        print(f"      text={str(r.text)[:60]!r}")
        # The region box must cover every line: mask refinement throws away mask
        # outside the regions, so a line left outside is never erased and the
        # original text stays visible under the translation.
        lines = getattr(r, "lines", None)
        if lines is not None:
            for j, ln in enumerate(lines):
                lb = aabb(ln)
                print(f"      line {j}: ({int(lb[0])},{int(lb[1])}) "
                      f"{int(lb[2]-lb[0])}x{int(lb[3]-lb[1])}")
        if args.save:
            cv2.imwrite(str(LAB_DIR / f"{args.name}_crop{i:02d}.png"),
                        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(LAB_DIR / f"{args.name}_th{i:02d}.png"), th)
    return 0


def cmd_mask(args) -> int:
    """Replay mask refinement offline from the fixture.

    The mask decides what inpainting erases, so "the balloon turned into
    artwork" is a mask bug, not a rendering bug - but the only way to tell which
    stage widened it is to run the stages separately on the same input.
    """
    import asyncio
    from manga_translator import mask_refinement

    with (LAB_DIR / f"{args.name}.pkl").open("rb") as fh:
        fx = pickle.load(fh)
    img_rgb = fx["img_rgb"]
    raw = fx.get("mask_raw")
    if raw is None:
        print("fixture has no mask_raw; regenerate it with the current fixture cmd")
        return 1
    regions = fx["text_regions"]
    print(f"mask_raw coverage={(raw > 0).mean():.4f}")

    seeded = 0
    seed_in = raw.copy()
    if not args.no_seed:
        seeded = mask_refinement.seed_unmasked_lines(seed_in, img_rgb, regions)
    print(f"seeded lines={seeded}  after-seed coverage={(seed_in > 0).mean():.4f}")

    # Stage-by-stage coverage inside one box. Guessing which stage widens the
    # mask wasted two rounds (the CRF and the dilation were both innocent), so
    # measure every step on the same pixels instead.
    if args.box:
        from manga_translator.mask_refinement.text_mask_utils import complete_mask
        from manga_translator.utils import Quadrilateral
        bx1, by1, bx2, by2 = (int(v) for v in args.box.split(","))

        def cov(mm, scale=1.0):
            a, b = int(bx1 * scale), int(by1 * scale)
            c, d = int(bx2 * scale), int(by2 * scale)
            return (mm[b:d, a:c] > 0).mean()

        print(f"[stage] mask_raw                {cov(raw):.3f}")
        print(f"[stage] after seeding           {cov(seed_in):.3f}")
        sf = max(min((seed_in.shape[0] - img_rgb.shape[0] / 3) / seed_in.shape[0], 1), 0.5)
        img_r = cv2.resize(img_rgb, (int(img_rgb.shape[1] * sf), int(img_rgb.shape[0] * sf)),
                           interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(seed_in, (int(img_rgb.shape[1] * sf), int(img_rgb.shape[0] * sf)),
                            interpolation=cv2.INTER_LINEAR)
        print(f"[stage] resized  (sf={sf:.3f})     {cov(mask_r, sf):.3f}   <- INTER_LINEAR")
        mask_r[mask_r > 0] = 255
        print(f"[stage] resized + threshold>0   {cov(mask_r, sf):.3f}")
        tls = [Quadrilateral(l * sf, '', 0) for r in regions for l in r.lines]
        fm = complete_mask(img_r, mask_r, tls, dilation_offset=args.dilation_offset,
                           kernel_size=args.kernel_size)
        print(f"[stage] complete_mask           {cov(fm, sf):.3f}")
        fm2 = cv2.resize(fm, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        fm2[fm2 > 0] = 255
        print(f"[stage] upscaled + threshold>0  {cov(fm2):.3f}")

    # complete_mask is the CRF + dilation step; call it the way dispatch does.
    out = asyncio.get_event_loop().run_until_complete(
        mask_refinement.dispatch(regions, img_rgb, seed_in, 'fit_text',
                                 dilation_offset=args.dilation_offset,
                                 kernel_size=args.kernel_size, verbose=False))
    print(f"refined coverage={(out > 0).mean():.4f}")
    tag = args.tag or ("noseed" if args.no_seed else "seed")
    cv2.imwrite(str(LAB_DIR / f"{args.name}_mask_{tag}.png"), out)

    # Per-region growth is what matters: a balloon eaten whole shows up as a
    # refined mask many times the area of the strokes it started from.
    print(f"{'#':>3} {'box':>24} {'raw%':>6} {'seed%':>6} {'refined%':>9}  text")
    for i, r in enumerate(regions):
        pts = np.array(r.min_rect).reshape(-1, 2)
        x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
        a = (raw[y1:y2, x1:x2] > 0).mean() if y2 > y1 and x2 > x1 else 0
        b = (seed_in[y1:y2, x1:x2] > 0).mean() if y2 > y1 and x2 > x1 else 0
        c = (out[y1:y2, x1:x2] > 0).mean() if y2 > y1 and x2 > x1 else 0
        print(f"{i:>3} {str((x1, y1, x2, y2)):>24} {a:>6.3f} {b:>6.3f} {c:>9.3f}  "
              f"{str(r.text)[:34]!r}")
    return 0


def cmd_flat(args) -> int:
    """Replay restore_flat_backgrounds on the fixture and crop the result.

    "fixed 6 areas" said nothing about the three balloons that stayed broken, and
    this is judged by pixels anyway. Writes original | inpainted | restored side
    by side for one box.
    """
    import logging

    from manga_translator.inpainting import flat_bg
    from manga_translator.inpainting.flat_bg import restore_flat_backgrounds

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    flat_bg.logger.setLevel(logging.DEBUG)
    flat_bg.logger.addHandler(logging.StreamHandler(sys.stdout))

    with (LAB_DIR / f"{args.name}.pkl").open("rb") as fh:
        fx = pickle.load(fh)
    img_rgb, mask, mask_raw = fx["img_rgb"], fx.get("mask"), fx.get("mask_raw")
    if mask is None:
        print("fixture has no mask; regenerate it")
        return 1
    before = fx["img_inpainted"].copy()
    after = before.copy()
    n = restore_flat_backgrounds(img_rgb, mask, after, fx["text_regions"], mask_raw)
    print(f"repainted lines={n} changed_px={(after != before).any(axis=2).sum()}")

    h, w = mask.shape[:2]
    boxes = []
    if args.box:
        boxes.append(tuple(int(v) for v in args.box.split(",")))
    for idx in [int(v) for v in args.regions.split(",")] if args.regions else []:
        pts = np.array(fx["text_regions"][idx].lines).reshape(-1, 2)
        pad = 40
        boxes.append((max(0, int(pts[:, 0].min()) - pad), max(0, int(pts[:, 1].min()) - pad),
                      min(w, int(pts[:, 0].max()) + pad), min(h, int(pts[:, 1].max()) + pad)))

    rows = []
    for x1, y1, x2, y2 in boxes:
        # The mask panel is what makes the others readable: "the balloon is white
        # here" means nothing until you can see whether that pixel was erased.
        mask_pan = cv2.cvtColor(((mask[y1:y2, x1:x2] > 0) * 255).astype(np.uint8),
                                cv2.COLOR_GRAY2RGB)
        sep = np.full((y2 - y1, 6, 3), 255, np.uint8)
        row = img_rgb[y1:y2, x1:x2]
        for pan in (mask_pan, before[y1:y2, x1:x2], after[y1:y2, x1:x2]):
            row = np.hstack([row, sep, pan])
        rows.append(row)
    width = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)),
                              constant_values=255) for r in rows])
    out = LAB_DIR / f"{args.name}_flat.png"
    cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"wrote {out} rows={len(rows)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe")
    p.add_argument("--name", required=True)
    p.add_argument("--save", action="store_true")
    p.set_defaults(func=cmd_probe)

    m = sub.add_parser("merge")
    m.add_argument("--name", required=True)
    m.add_argument("--fs-tol", type=float, default=1.3)
    m.add_argument("--ar-tol", type=float, default=1.3)
    m.add_argument("--gap1", type=float, default=1.0)
    m.add_argument("--gap2", type=float, default=3.0)
    m.add_argument("--slack", type=float, default=1.35)
    m.set_defaults(func=cmd_merge)

    f = sub.add_parser("fixture")
    f.add_argument("--url", default="")
    f.add_argument("--path", default="", help="local image path instead of --url")
    f.add_argument("--name", required=True)
    f.add_argument("--detector", default="ctd")
    f.add_argument("--ocr", default="48px_ctc")
    f.add_argument("--target-lang", default="CHS")
    f.add_argument("--strategy", default="cascade")
    f.add_argument("--text-th", type=float, default=0.3)
    f.add_argument("--box-th", type=float, default=0.4)
    f.add_argument("--unclip", type=float, default=2.8)
    f.set_defaults(func=cmd_fixture)

    k = sub.add_parser("mask")
    k.add_argument("--name", required=True)
    k.add_argument("--no-seed", action="store_true",
                   help="skip seed_unmasked_lines, to tell whether seeding is the cause")
    k.add_argument("--dilation-offset", type=int, default=0)
    k.add_argument("--kernel-size", type=int, default=3)
    k.add_argument("--box", default="", help="x1,y1,x2,y2 to get per-stage coverage")
    k.add_argument("--tag", default="")
    k.set_defaults(func=cmd_mask)

    fl = sub.add_parser("flat")
    fl.add_argument("--name", required=True)
    fl.add_argument("--box", default="", help="x1,y1,x2,y2")
    fl.add_argument("--regions", default="", help="region indices, e.g. 4,11,12")
    fl.set_defaults(func=cmd_flat)

    r = sub.add_parser("render")
    r.add_argument("--name", required=True)
    r.add_argument("--strategy", default="cascade")
    r.add_argument("--translations", default="")
    r.add_argument("--only", type=int, default=-1)
    r.set_defaults(func=cmd_render)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
