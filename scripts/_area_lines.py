"""List the line boxes and raw detections that fall in an area of a fixture.

An erased rectangle on the page has to be traced back to whichever line box
produced it before it can be fixed, and the box that looks responsible often is
not (the one next to it is).

  python scripts/_area_lines.py mk3 300,1480,430,1600
"""
import pickle
import sys

import numpy as np

name, box = sys.argv[1], sys.argv[2]
bx1, by1, bx2, by2 = (int(v) for v in box.split(","))
fx = pickle.load(open(f"/app/render_lab/{name}.pkl", "rb"))


def aabb(pts):
    p = np.array(pts).reshape(-1, 2)
    return int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())


def hits(b):
    return b[0] < bx2 and b[2] > bx1 and b[1] < by2 and b[3] > by1


for i, r in enumerate(fx["text_regions"]):
    for j, ln in enumerate(r.lines):
        b = aabb(ln)
        if hits(b):
            print(f"r{i} line{j} {b} font={r.font_size} text={r.text[:24]!r}")

print("--- raw textlines ---")
for t in fx.get("textlines") or []:
    b = aabb(t.pts)
    if hits(b):
        print(f"det {b} text={getattr(t, 'text', None)!r}")
