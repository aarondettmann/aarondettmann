#!/usr/bin/env python3
# Copyright 2026 Aaron Dettmann. All rights reserved.
"""Generate the halftone moon avatar as SVG files.

The moon is a shaded sphere, printed in one ink on one paper color with a 45 degree
halftone screen. The sky is a solid ink field; the moon is a paper disc printed over it.
Dots that cross the limb print ink on ink, so the limb remains sharp. A smoothstep of the
illumination gives a uniform lit area and a graded terminator.

The maria, the large dark plains of the near side, are placed at approximate
selenographic coordinates. Each mare is a soft disc on the sphere and is therefore
foreshortened toward the limb. Low-frequency Perlin noise roughens its boundary.

An optional earthshine term prints the night side as a light tint.

Usage (requires numpy):

    python3 moon_avatar.py [output-directory]

This writes moon.svg and moon-earthshine.svg, by default next to this script. The mare
boundaries depend on numpy's seeded Generator, whose stream may change between numpy
releases. To export a PNG with the copyright notice (requires Inkscape and Pillow), run in
this directory:

    inkscape moon.svg -o moon.png -w 1024 --export-background-opacity=1
    python3 -c "import moon_avatar; moon_avatar.tag_png('moon.png')"
"""

import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

SIZE = 512
COPYRIGHT = "Copyright 2026 Aaron Dettmann. All rights reserved."

# Approximate centers and angular radii of the main near-side maria:
# (longitude east, latitude north, radius), all in degrees.
# fmt: off
MARIA = [
    (-57, 18, 24),   # Oceanus Procellarum
    (-16, 33, 15),   # Mare Imbrium
    (18, 28, 9),     # Mare Serenitatis
    (31, 8, 11),     # Mare Tranquillitatis
    (59, 17, 7),     # Mare Crisium
    (51, -8, 10),    # Mare Fecunditatis
    (35, -15, 5),    # Mare Nectaris
    (-17, -21, 10),  # Mare Nubium
    (-39, -24, 5),   # Mare Humorum
]
# fmt: on
# Mare Frigoris is omitted: at one to two screen cells wide it prints as a dashed line.

# Mare radius, in degrees, at which the boundary noise has its nominal amplitude.
REFERENCE_RADIUS = 15


def unit(lon, lat):
    """Return the unit vector for a longitude and latitude in degrees (axes of maria_cover)."""
    lo, la = math.radians(lon), math.radians(lat)
    return math.cos(la) * math.sin(lo), -math.sin(la), math.cos(la) * math.cos(lo)


def perlin(x, y, seed=0):
    """2D gradient noise on coordinate arrays, with values in about [-0.7, 0.7]."""
    rng = np.random.default_rng(seed)
    perm = np.tile(rng.permutation(256), 2)
    ang = rng.uniform(0, 2 * np.pi, 256)
    gx, gy = np.cos(ang), np.sin(ang)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    xi, yi = x0 & 255, y0 & 255

    def grad(ix, iy, dx, dy):
        h = perm[perm[ix] + iy]
        return gx[h] * dx + gy[h] * dy

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    u, v = fade(fx), fade(fy)
    n00 = grad(xi, yi, fx, fy)
    n10 = grad(xi + 1, yi, fx - 1, fy)
    n01 = grad(xi, yi + 1, fx, fy - 1)
    n11 = grad(xi + 1, yi + 1, fx - 1, fy - 1)
    return (n00 * (1 - u) + n10 * u) * (1 - v) + (n01 * (1 - u) + n11 * u) * v


def smoothstep(e0, e1, v):
    """Cubic Hermite step from 0 at v <= e0 to 1 at v >= e1. Requires e0 < e1."""
    s = np.clip((v - e0) / (e1 - e0), 0, 1)
    return s * s * (3 - 2 * s)


def dot_radius(ink):
    """Dot radius, in units of the screen spacing, that covers the fraction ink of the paper.

    Up to radius 1/sqrt(2) each dot overlaps only its four neighbors, two lenses per cell,
    and the coverage reaches 1. The coverage is inverted by table interpolation. At ink 1
    the radius 0.75 exceeds 1/sqrt(2), so solid areas print without specks where four dots
    meet, at output widths of about 256 px and above. The radius jumps to 0.75 only at ink 1.
    """
    r = np.linspace(0, 1 / math.sqrt(2), 2001)
    h = np.clip(0.5 / np.maximum(r, 1e-12), -1, 1)
    lens = 2 * r * r * np.arccos(h) - 0.5 * np.sqrt(np.clip(4 * r * r - 1, 0, None))
    cover = np.pi * r * r - 2 * lens
    return np.where(ink < 1, np.interp(ink, cover, r), 0.75)


def maria_cover(x, y, z, roughness=9.0, seed=7):
    """Coverage of the maria, in 0..1, at points (x, y, z) on the visible unit hemisphere.

    x points right (east), y down (south) and z toward the viewer. roughness scales the
    boundary noise: the displacement peaks at about 0.7 times roughness degrees for a mare
    of radius REFERENCE_RADIUS and is proportional to the radius.
    """
    lon = np.degrees(np.arctan2(x, z))
    lat = np.degrees(np.arcsin(np.clip(-y, -1, 1)))
    wobble = roughness * perlin(lon / 14 + 3.1, lat / 14 + 1.7, seed)
    cover = np.zeros_like(x)
    for lo, la, rad in MARIA:
        cx, cy, cz = unit(lo, la)
        dist = np.degrees(np.arccos(np.clip(x * cx + y * cy + z * cz, -1, 1)))
        shift = wobble * rad / REFERENCE_RADIUS
        cover = np.maximum(cover, 1 - smoothstep(rad * 0.45, rad * 1.35, dist + shift))
    return cover


def moon_svg(
    ink="#0c1f3b",
    paper="#f1e9d4",
    spacing=8.6,
    radius=170.0,
    center=(232.0, 262.0),
    light=(0.78, -0.36, 0.52),
    edge=(0.16, 0.5),
    maria=0.33,
    earthshine=0.0,
    angle=45,
    min_dot=0.5,
    title="Halftone moon",
):
    """Return the avatar as an SVG document. The screen grid is anchored at the moon's center.

    light: direction toward the sun, in the axes of maria_cover.
    edge: illumination values (cosine of incidence) at which the terminator starts and ends.
    maria: relative darkening at full maria coverage.
    min_dot: dot radius in SVG units at or below which dots are omitted.
    """
    cx, cy = center
    R = radius
    lx, ly, lz = np.array(light) / np.linalg.norm(light)

    n = math.ceil(R / spacing) + 1
    g = np.arange(-n, n + 1) * spacing
    U, V = np.meshgrid(g, g)
    a = math.radians(angle)
    X = cx + U * math.cos(a) - V * math.sin(a)
    Y = cy + U * math.sin(a) + V * math.cos(a)
    d = np.hypot(U, V)
    keep = d < R + spacing
    X, Y, d = X[keep], Y[keep], d[keep]

    x, y = (X - cx) / R, (Y - cy) / R
    z = np.sqrt(np.clip(1 - x * x - y * y, 0, 1))
    day = smoothstep(edge[0], edge[1], lx * x + ly * y + lz * z)
    b = earthshine + (1 - earthshine) * day
    b = b * (1 - maria * maria_cover(x, y, z))

    # Dots that cross the limb print ink on ink; a clip would add a second antialiased edge at
    # the limb. Dots that lie entirely outside the disc are omitted.
    r = spacing * dot_radius(1 - b)
    dots = "".join(
        f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{q:.2f}"/>'
        for px, py, q, dist in zip(X, Y, r, d)
        if q > min_dot and dist - q < R
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" '
        f'width="{SIZE}" height="{SIZE}">\n'
        f"  <!-- {COPYRIGHT} -->\n"
        f"  <title>{escape(title)}</title>\n"
        f'  <rect width="{SIZE}" height="{SIZE}" fill="{ink}"/>\n'
        f'  <circle cx="{cx}" cy="{cy}" r="{R}" fill="{paper}"/>\n'
        f'  <g fill="{ink}">{dots}</g>\n'
        f"</svg>\n"
    )


def tag_png(path):
    """Store the copyright notice in a PNG and remove its alpha channel. Requires Pillow."""
    from PIL import Image, PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("Copyright", COPYRIGHT)
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    rgb.save(path, optimize=True, pnginfo=info)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "moon.svg").write_text(moon_svg(), encoding="utf-8")
    # With earthshine the whole disc shows, so it is centered.
    (out / "moon-earthshine.svg").write_text(
        moon_svg(earthshine=0.05, center=(256.0, 256.0), title="Halftone moon with earthshine"),
        encoding="utf-8",
    )
