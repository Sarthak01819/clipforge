"""Generate assets/clipforge.ico (and a PNG) with numpy + ffmpeg - no image libraries needed.

The icon is a rounded purple square with a white "play" triangle cut by a diagonal
slash (a clip being cut). Run:  .venv\\Scripts\\python tools\\make_icon.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.ffmpeg_utils import require_ffmpeg  # noqa: E402

OUT_DIR = ROOT / "assets"
SIZE = 1024  # draw large, downscale with ffmpeg for smooth edges


def _render(n: int) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    img = np.zeros((n, n, 4), dtype=np.float32)

    # Rounded square background with a diagonal purple gradient.
    r = 0.22
    dx = np.maximum(np.abs(x - 0.5) - (0.5 - r), 0)
    dy = np.maximum(np.abs(y - 0.5) - (0.5 - r), 0)
    inside = np.sqrt(dx ** 2 + dy ** 2) <= r - 0.004
    t = (x + y) / 2
    c1, c2 = np.array([0x5b, 0x3c, 0xf0]), np.array([0xa4, 0x8b, 0xff])
    img[..., :3] = (c1 * (1 - t)[..., None] + c2 * t[..., None]) / 255
    img[..., 3] = inside

    # White play triangle.
    ax, ay, bx, by, cx, cy = 0.36, 0.26, 0.36, 0.74, 0.76, 0.50

    def edge(px, py, qx, qy):
        return (qx - px) * (y - py) - (qy - py) * (x - px)

    e1, e2, e3 = edge(ax, ay, bx, by), edge(bx, by, cx, cy), edge(cx, cy, ax, ay)
    tri = ((e1 <= 0) & (e2 <= 0) & (e3 <= 0)) | ((e1 >= 0) & (e2 >= 0) & (e3 >= 0))

    # Diagonal "cut" gap through the triangle.
    cut = np.abs((x - 0.5) + (y - 0.5) * 0.9 - 0.02) < 0.035
    white = tri & ~cut & inside
    img[white, :3] = 1.0
    return (img * 255).astype(np.uint8)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    big = OUT_DIR / "clipforge_1024.rgba"
    big.write_bytes(_render(SIZE).tobytes())
    ffmpeg = require_ffmpeg()
    raw = ["-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{SIZE}x{SIZE}", "-i", str(big)]
    png = OUT_DIR / "clipforge.png"
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *raw,
                    "-vf", "scale=256:256:flags=lanczos", str(png)], check=True)
    # Multi-resolution .ico: one stream per size.
    sizes = [256, 64, 48, 32, 16]
    split = f"split={len(sizes)}" + "".join(f"[s{i}]" for i in range(len(sizes)))
    scales = ";".join(f"[s{i}]scale={s}:{s}:flags=lanczos[o{i}]" for i, s in enumerate(sizes))
    maps = [a for i in range(len(sizes)) for a in ("-map", f"[o{i}]")]
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *raw,
                    "-filter_complex", f"[0:v]{split};{scales}", *maps, "-c:v", "png",
                    str(OUT_DIR / "clipforge.ico")], check=True)
    big.unlink()
    print("wrote", OUT_DIR / "clipforge.ico", "and", png)


if __name__ == "__main__":
    main()
