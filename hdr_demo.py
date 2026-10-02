#!/usr/bin/env python3
"""Procedural 30 s HDR10 (PQ / BT.2020) demo video for YouTube.

Every frame is rendered in linear light with absolute units (nits, cd/m^2) in
BT.709 primaries, converted to BT.2020, PQ-encoded, turned into 10-bit
limited-range Y'CbCr 4:2:0 and piped straight into x265.

  python3 hdr_demo.py stats     # measure MaxCLL/MaxFALL + write SDR preview stills
  python3 hdr_demo.py encode    # render + encode hdr_demo.mp4
"""
import json, math, os, subprocess, sys
from functools import lru_cache
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile
from scipy.signal import fftconvolve

W, H, FPS, DUR = 1920, 1080, 30, 30.0
NFRAMES = int(FPS * DUR)
PEAK = 4000.0          # mastering peak, nits
TEXT_NITS = 203.0      # BT.2408 reference "graphics white"
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("HDR_SCRATCH", HERE)

# scene boundaries (seconds)
S1, S2, S3, S4 = 0.0, 8.0, 16.0, 23.5

M709_2020 = np.array([[0.6274, 0.3293, 0.0433],
                      [0.0691, 0.9195, 0.0114],
                      [0.0164, 0.0880, 0.8956]], np.float32)

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def col(*c):
    return np.array(c, np.float32)


def pq_encode(L):
    y = np.clip(L / 10000.0, 0.0, 1.0)
    m1, m2 = 2610 / 16384, 2523 / 4096 * 128
    c1, c2, c3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    yp = np.power(y, m1)
    return np.power((c1 + c2 * yp) / (1 + c3 * yp), m2)


def pq_decode(E):
    m1, m2 = 2610 / 16384, 2523 / 4096 * 128
    c1, c2, c3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    ep = np.power(np.clip(E, 0, 1), 1 / m2)
    return 10000.0 * np.power(np.maximum(ep - c1, 0) / (c2 - c3 * ep), 1 / m1)


def noise(shape_small, shape_big, rng, interp=cv2.INTER_CUBIC):
    n = rng.random(shape_small).astype(np.float32)
    return cv2.resize(n, (shape_big[1], shape_big[0]), interpolation=interp)


def bloom(img, gain=0.06, sigma=6):
    small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    b = cv2.GaussianBlur(small, (0, 0), sigma)
    b2 = cv2.GaussianBlur(small, (0, 0), sigma * 4)
    up = cv2.resize(b + 0.5 * b2, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return img + gain * up


# ----------------------------------------------------------------------------
# text
@lru_cache(maxsize=4096)
def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


@lru_cache(maxsize=4096)
def text_mask(s, size, bold=False, stroke=0):
    f = font(size, bold)
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    l, t, r, b = d.textbbox((0, 0), s, font=f, stroke_width=stroke)
    pad = 8 + stroke
    im = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad))
    ImageDraw.Draw(im).text((pad - l, pad - t), s, font=f, fill=255,
                            stroke_width=stroke, stroke_fill=255)
    m = np.asarray(im, np.float32) / 255.0
    return m, pad


def add_mask(img, m, x, y, rgb):
    """Add mask*rgb at top-left (x,y), clipped to img."""
    h, w = m.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    sub = m[y0 - y:y1 - y, x0 - x:x1 - x]
    img[y0:y1, x0:x1] += sub[..., None] * rgb


def mul_mask(img, m, x, y, k):
    h, w = m.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    sub = m[y0 - y:y1 - y, x0 - x:x1 - x]
    img[y0:y1, x0:x1] *= (1 - k * sub)[..., None]


@lru_cache(maxsize=4096)
def shadow_mask(s, size, bold):
    m, pad = text_mask(s, size, bold, stroke=3)
    return cv2.GaussianBlur(m, (0, 0), 3)


def put_text(img, s, x, y, size, nits=TEXT_NITS, rgb=(1, 1, 1), anchor="l",
             bold=False, alpha=1.0, shadow=True):
    if alpha <= 0.002:
        return
    m, pad = text_mask(s, size, bold)
    w = m.shape[1] - 2 * pad
    if anchor == "c":
        x -= w // 2
    elif anchor == "r":
        x -= w
    x, y = int(x) - pad, int(y) - pad
    if shadow:
        mul_mask(img, shadow_mask(s, size, bold), x, y, 0.85 * alpha)
    add_mask(img, m, x, y, col(*rgb) * nits * alpha)


def leader(img, p0, p1, nits, alpha):
    if alpha <= 0.002:
        return
    m = np.zeros((H, W), np.float32)
    cv2.line(m, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 1.0, 2, cv2.LINE_AA)
    cv2.circle(m, (int(p0[0]), int(p0[1])), 4, 1.0, 1, cv2.LINE_AA)
    sh = cv2.GaussianBlur(m, (0, 0), 2.5)
    img *= (1 - 0.8 * alpha * np.clip(sh * 2, 0, 1))[..., None]
    img += (m * nits * alpha)[..., None]


def callout(img, label, p, offset, alpha, nits=TEXT_NITS):
    ox = offset[0]
    if (ox > 0 and p[0] + ox > W - 420) or (ox < 0 and p[0] + ox < 420):
        ox = -ox
    offset = (ox, offset[1])
    tx, ty = p[0] + offset[0], p[1] + offset[1]
    leader(img, p, (tx, ty), nits * 0.8, alpha)
    anchor = "l" if offset[0] >= 0 else "r"
    put_text(img, label, tx + (10 if offset[0] >= 0 else -10), ty - 16, 30,
             nits=nits, anchor=anchor, alpha=alpha)


def scene_alpha(t, a, b, fade=0.6):
    return float(smoothstep(a, a + fade, t) * (1 - smoothstep(b - fade, b, t)))


# ----------------------------------------------------------------------------
# Scene 1: sunset over water
_r = np.random.default_rng(11)
CLOUD_W = W + 600
_cn = (0.55 * noise((10, 30), (H, CLOUD_W), _r) + 0.3 * noise((22, 70), (H, CLOUD_W), _r)
       + 0.15 * noise((60, 260), (H, CLOUD_W), _r))
_cy = np.arange(H, dtype=np.float32)[:, None]
_band = (np.exp(-((_cy - 0.20 * H) / (0.05 * H)) ** 2) + 0.9 * np.exp(-((_cy - 0.36 * H) / (0.035 * H)) ** 2)
         + 0.6 * np.exp(-((_cy - 0.50 * H) / (0.018 * H)) ** 2))
CLOUDS = (np.clip((_cn - 0.47) * 4.0, 0, 1) * np.clip(_band, 0, 1)).astype(np.float32)
YH = 0.64 * H  # horizon


def headland_profile():
    x = np.arange(W, dtype=np.float32)
    r = np.random.default_rng(5)
    bumps = sum(a * np.sin(x / p + r.random() * 6) for a, p in [(14, 37), (8, 13), (25, 90)])
    base = (YH + 40 - 260 * smoothstep(0.68 * W, 0.86 * W, x) + 60 * smoothstep(0.9 * W, W, x)
            + 520 * (1 - smoothstep(0.58 * W, 0.70 * W, x)))
    prof = base + bumps * smoothstep(0.60 * W, 0.72 * W, x)
    return prof.astype(np.float32)


HEADLAND = headland_profile()
ROCK_TEX = (0.6 + 0.8 * (0.6 * noise((70, 120), (H, W), _r) + 0.4 * noise((300, 500), (H, W), _r))).astype(np.float32)


def sun_pos(tt):
    return 0.40 * W + tt * 5.0, 0.30 * H + (tt / 8.0) * 0.27 * H


def sky_at(X, Y, tt):
    sx, sy = sun_pos(tt)
    late = tt / 8.0
    v = np.clip((YH - Y) / YH, 0, 1)[..., None]
    horizon = col(1.0, 0.40, 0.12) * (220 - 70 * late)
    mid = col(0.85, 0.32, 0.38) * (55 - 15 * late)
    top = col(0.10, 0.18, 0.50) * (14 - 5 * late)
    sky = np.where(v < 0.35, horizon + (mid - horizon) * (v / 0.35),
                   mid + (top - mid) * np.maximum((v - 0.35) / 0.65, 0) ** 0.7)
    d = np.sqrt((X - sx) ** 2 + (Y - sy) ** 2)
    glow = (1600 * np.exp(-d / 35) + 420 * np.exp(-d / 140) + 70 * np.exp(-d / 480))[..., None]
    sky = sky + glow * col(1.0, 0.62, 0.32)
    R = 46.0
    disk = np.clip((R - d) / 1.5, 0, 1)
    limb = 1 - 0.35 * np.clip(d / R, 0, 1) ** 2
    sun = (disk * limb)[..., None] * col(1.0, 0.90 - 0.12 * late, 0.74 - 0.2 * late) * PEAK
    sky = sky * (1 - disk[..., None]) + sun
    # clouds drift right; lit by the sun (silver linings near it)
    c = cv2.remap(CLOUDS, (X + 300 - tt * 9).astype(np.float32), Y.astype(np.float32),
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)[..., None]
    lit = col(1.0, 0.50, 0.38) * (12 + 50 * (1 - v)) + (900 * np.exp(-d / 200))[..., None] * col(1, 0.75, 0.5)
    return sky * (1 - 0.92 * c) + lit * c


def render_sunset(tt, fi):
    img = np.empty((H, W, 3), np.float32)
    ys = int(math.ceil(YH))
    img[:ys] = sky_at(XX[:ys], YY[:ys], tt)
    # water: mirrored, rippled, perspective-compressed sky
    Xw, Yw = XX[ys:], YY[ys:]
    depth = (Yw - YH) / (H - YH)
    ripple_x = 7 * np.sin(Yw * 0.21 + tt * 2.1) * np.sin(Xw * 0.013 + Yw * 0.05 + tt * 1.3)
    ym = YH - (Yw - YH) * 1.25 + 2.5 * np.sin(Yw * 0.6 + tt * 3.0)
    refl = sky_at(Xw + ripple_x, ym, tt) * (0.55 - 0.35 * depth)[..., None]
    water = col(0.02, 0.05, 0.10) * 6 + refl
    sx, sy = sun_pos(tt)
    rng = np.random.default_rng(1000 + fi)
    n = noise(((H - ys) // 3, W // 14), (H - ys, W), rng, cv2.INTER_LINEAR)
    sp = np.clip((n - 0.80) / 0.20, 0, 1) ** 2
    width = 18 + depth * 420
    env = np.exp(-((Xw - sx) / width) ** 2) * np.exp(-depth * 0.8)
    water += (env * (sp * 3600 + 120))[..., None] * col(1.0, 0.80, 0.55)
    img[ys:] = water
    # sailboat silhouette crossing the glitter path
    bx, by = 0.36 * W + tt * 22, YH + 70
    m = np.zeros((H, W), np.float32)
    pts_s = np.array([[bx, by - 8], [bx + 4, by - 120], [bx + 58, by - 10]], np.int32)
    pts_h = np.array([[bx - 30, by - 8], [bx + 75, by - 8], [bx + 60, by + 6], [bx - 18, by + 6]], np.int32)
    pts_j = np.array([[bx - 4, by - 10], [bx - 2, by - 100], [bx - 40, by - 12]], np.int32)
    for p in (pts_s, pts_h, pts_j):
        cv2.fillPoly(m, [p], 1.0, cv2.LINE_AA)
    cv2.line(m, (int(bx + 2), int(by - 6)), (int(bx + 2), int(by - 128)), 1.0, 2, cv2.LINE_AA)
    img = img * (1 - m[..., None]) + (m * 0.015)[..., None] * col(1, 0.9, 0.8)
    # headland: deep shadow with visible texture, faint warm rim
    prof = HEADLAND[None, :]
    inside = np.clip(YY - prof, 0, 1)
    rim = np.exp(-np.maximum(YY - prof, 0) / 4) * inside
    rock = (0.035 * ROCK_TEX * (1 + 2.5 * np.clip((YY - prof) / 600, 0, 1)))[..., None] * col(0.9, 0.8, 0.75)
    rock = rock + (rim * 9)[..., None] * col(1, 0.55, 0.3)
    img = img * (1 - inside[..., None]) + rock * inside[..., None]
    img = bloom(img, 0.05)
    return img


def labels_sunset(img, tt, a):
    sx, sy = sun_pos(tt)
    put_text(img, "1  Sunset over water", 80, 70, 44, bold=True, alpha=a)
    put_text(img, "The sun is 40× brighter than SDR white, and the rocks stay visible",
             80, 128, 28, nits=170, alpha=a)
    ca = a * float(smoothstep(1.2, 2.0, tt))
    callout(img, "Sun disk ≈ 4,000 nits", (sx - 40, sy - 40), (-130, -110), ca)
    callout(img, "Sun glitter: up to 3,600 nits", (sx + 40, YH + 200), (200, -60), ca * float(smoothstep(2.2, 3.0, tt)))
    callout(img, "Shadowed rocks ≈ 0.03–0.1 nits", (0.86 * W, 0.66 * H), (-150, -170), ca * float(smoothstep(3.2, 4.0, tt)))


# ----------------------------------------------------------------------------
# Scene 2: fireworks
_fr = np.random.default_rng(7)
FW_COLORS = [col(1.0, 0.12, 0.08), col(1.0, 0.65, 0.25), col(0.25, 1.0, 0.35),
             col(0.30, 0.45, 1.0), col(1, 1, 1), col(1.0, 0.25, 0.9), col(0.3, 0.9, 1.0)]
BURST_T = [0.5, 1.3, 1.9, 2.6, 3.2, 3.9, 4.5, 5.0, 5.4, 5.8, 6.1, 6.35]
BURSTS = []
for i, t0 in enumerate(BURST_T):
    n = 420
    v3 = _fr.normal(size=(n, 3))
    v3 /= np.linalg.norm(v3, axis=1, keepdims=True)
    BURSTS.append(dict(
        t0=t0, x=float(_fr.uniform(0.18, 0.82) * W), y=float(_fr.uniform(0.18, 0.42) * H),
        c=FW_COLORS[i % len(FW_COLORS)], c2=FW_COLORS[(i * 3 + 2) % len(FW_COLORS)],
        dirs=v3[:, :2].astype(np.float32), spd=(_fr.uniform(0.85, 1.0, n) * _fr.uniform(330, 470)).astype(np.float32),
        life=_fr.uniform(0.55, 0.85), crackle=bool(i % 3 == 2), seed=i, x0=float(_fr.uniform(-60, 60))))
FW_GROUND = (0.86 * H + 18 * np.sin(np.arange(W) / 140.0) + 10 * np.sin(np.arange(W) / 47.0 + 1)
             + 22 * (noise((1, 90), (1, W), _fr, cv2.INTER_CUBIC)[0] - 0.5)).astype(np.float32)
_stars = np.zeros((H, W, 3), np.float32)
for _ in range(500):
    x, y = int(_fr.uniform(0, W)), int(_fr.uniform(0, 0.85 * H))
    _stars[y, x] += float(np.exp(_fr.normal(-1.5, 1.0))) * 6 * col(0.85, 0.9, 1.0)
STARS = cv2.GaussianBlur(_stars, (0, 0), 0.7)
FW_SKY = ((0.003 + 0.03 * smoothstep(0.3 * H, 0.86 * H, YY) ** 2)[..., None] * col(0.6, 0.7, 1.0)).astype(np.float32)


def splat(buf, px, py, vals):
    ix, iy = np.round(px).astype(np.int32), np.round(py).astype(np.int32)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    np.add.at(buf, (iy[ok], ix[ok]), vals[ok])


def render_fireworks(tt, fi):
    buf = np.zeros((H, W, 3), np.float32)
    glow = np.zeros((H, W, 3), np.float32)
    ground_light = np.zeros(3, np.float32)
    k, g = 1.6, 70.0
    for b in BURSTS:
        tau = tt - b["t0"]
        if -0.9 < tau < 0:  # rising rocket
            for j in range(40):
                tj = tau - j * 0.006
                if tj < -0.9:
                    break
                u = (tj + 0.9) / 0.9
                e = 1 - (1 - u) ** 2
                px = b["x"] + b["x0"] * (1 - e)
                py = 0.92 * H + (b["y"] - 0.92 * H) * e
                splat(buf, np.array([px]), np.array([py]), (col(1, 0.7, 0.35) * 500 * math.exp(-j / 12))[None])
        if 0 <= tau < 3.6:
            n = len(b["spd"])
            idx = np.arange(n)
            for j in range(28):
                tj = tau - j * 0.009
                if tj < 0:
                    break
                disp = (b["spd"] * (1 - math.exp(-k * tj)) / k)[:, None] * b["dirs"]
                px = b["x"] + disp[:, 0]
                py = b["y"] + disp[:, 1] + 0.5 * g * tj * tj
                bright = 1500 * math.exp(-tj / b["life"]) * math.exp(-j / 7.0)
                mix = min(tj / 1.0, 1.0)
                c = b["c"] * (1 - mix) + b["c2"] * mix
                amp = np.full(n, bright, np.float32)
                if b["crackle"] and tj > 0.9:
                    h = ((idx * 7919 + int(tj * 18) * 104729 + b["seed"] * 31) % 1000) / 1000.0
                    amp = np.where(h > 0.82, 2600 * math.exp(-(tj - 0.9) / 1.2), 0).astype(np.float32) * math.exp(-j / 3.5)
                    c = col(1, 0.95, 0.85)
                splat(buf, px, py, amp[:, None] * c[None])
            ground_light += b["c"] * 6.0 * math.exp(-tau / 0.45)
            if tau < 0.3:
                d = np.sqrt((XX - b["x"]) ** 2 + (YY - b["y"]) ** 2)
                glow += (np.exp(-d / 45) * 1500 * (1 - tau / 0.3) ** 2)[..., None] * col(1, 0.95, 0.85)
                ground_light += col(1, 0.95, 0.9) * 25 * (1 - tau / 0.3)
    core = np.minimum(cv2.GaussianBlur(buf, (0, 0), 1.0) * 6.0 + glow, PEAK)
    img = FW_SKY + STARS + core
    img = bloom(img, 0.04, 5)
    # hills/treeline lit by the bursts
    inside = np.clip(YY - FW_GROUND[None, :], 0, 1)[..., None]
    hill = (0.002 + ground_light[None, None, :] * 0.25 * (1 - smoothstep(0.86 * H, H, YY))[..., None]
            * (0.7 + 0.6 * ROCK_TEX[..., None]))
    return img * (1 - inside) + hill * inside


def labels_fireworks(img, tt, a):
    put_text(img, "2  Fireworks at night", 80, 70, 44, bold=True, alpha=a)
    put_text(img, "Thousands of brilliant sparks against a sky that is almost black",
             80, 128, 28, nits=170, alpha=a)
    ca = a * float(smoothstep(1.0, 1.6, tt))
    b = BURSTS[1]
    callout(img, "Sparks up to 4,000 nits", (b["x"], b["y"] + 170), (220, 120), ca * float(1 - smoothstep(2.6, 3.0, tt)))
    callout(img, "Night sky ≈ 0.003 nits", (0.06 * W, 0.62 * H), (60, 70), ca)
    callout(img, "Hills lit by the flashes", (0.30 * W, 0.93 * H), (80, -40), a * float(smoothstep(3.0, 3.6, tt)))


# ----------------------------------------------------------------------------
# Scene 3: night city
PAN = 760
CW = W + PAN + 2
YB = int(0.80 * H)   # building base
YR = YB + 44         # river edge


SIGNS = [("HOTEL", 260, int(0.42 * H), 64, col(1.0, 0.1, 0.55), 2200, 0),
         ("OPEN 24h", 1150, int(0.60 * H), 44, col(0.1, 0.9, 1.0), 1800, 1),
         ("HDR", 1800, int(0.45 * H), 90, col(1.0, 0.35, 0.05), 2600, 2),
         ("BAR", 2400, int(0.56 * H), 52, col(0.4, 1.0, 0.25), 1600, 0)]


def build_city():
    r = np.random.default_rng(3)
    E = np.zeros((H, CW, 3), np.float32)
    yv = np.arange(H, dtype=np.float32)[:, None, None]
    E[:] = col(0.25, 0.30, 0.55) * 0.008 + col(1.0, 0.55, 0.25) * 0.5 * smoothstep(0.35 * H, YB, yv) ** 3
    for _ in range(140):
        x, y = int(r.uniform(0, CW)), int(r.uniform(0, 0.45 * H))
        E[y, x] += float(np.exp(r.normal(-1, 0.7))) * 3
    # moon
    mx, my, mr = int(0.25 * W + PAN * 0.9), int(0.17 * H), 36
    Yg, Xg = np.mgrid[0:H, 0:CW].astype(np.float32)
    d = np.sqrt((Xg - mx) ** 2 + (Yg - my) ** 2)
    crater = 0.82 + 0.25 * noise((9, 9 * CW // H), (H, CW), r)
    E += (np.clip((mr - d) / 1.5, 0, 1) * crater * 1100)[..., None] * col(0.95, 0.95, 1.0)
    E += ((30 * np.exp(-d / 90) + 4 * np.exp(-d / 400)) * (d > mr))[..., None] * col(0.8, 0.85, 1.0)
    toggles, beacons, crowns = [], [], []

    def buildings(far):
        x = -int(r.uniform(0, 80))
        while x < CW:
            bw = int(r.uniform(70, 200) if not far else r.uniform(50, 130))
            hgt = int(r.uniform(150, 600) if not far else r.uniform(200, 470))
            top = YB - hgt
            body = (col(0.35, 0.4, 0.55) * 0.07) if far else (col(0.6, 0.55, 0.5) * 0.03)
            grad = (1 + 3 * smoothstep(top, YB, np.arange(top, YB, dtype=np.float32)))[:, None, None]
            E[top:YB, max(x, 0):max(x + bw, 0)] = body * grad * (0.8 + 0.4 * r.random())
            cw_, ch_, ww, wh = (9, 13, 4, 7) if far else (15, 21, 8, 12)
            for cy in range(top + 10, YB - 16, ch_):
                for cx in range(x + 6, x + bw - ww - 4, cw_):
                    if cx < 0 or cx + ww >= CW:
                        continue
                    if r.random() < (0.30 if far else 0.38):
                        nits = float(np.exp(r.normal(math.log(35), 0.8)))
                        nits = min(nits, 300) * (0.35 if far else 1.0)
                        hue = r.choice(3, p=[0.6, 0.25, 0.15])
                        c = [col(1.0, 0.72, 0.42), col(0.75, 0.88, 1.0), col(1, 1, 0.95)][hue]
                        if not far and r.random() < 0.04:
                            toggles.append((cx, cy, ww, wh, c * nits, r.uniform(0, 10), r.uniform(0.2, 0.7)))
                        else:
                            E[cy:cy + wh, cx:cx + ww] = c * nits
                    else:
                        E[cy:cy + wh, cx:cx + ww] *= 0.6
            if not far and hgt > 470:
                beacons.append((x + bw // 2, top - 6, r.uniform(0, 1)))
                E[top - 6:top, x + bw // 2 - 1:x + bw // 2 + 1] = 0.02
            if not far and r.random() < 0.18:
                crowns.append((x, top, bw))
                E[top:top + 3, max(x, 0):max(x + bw, 0)] = col(0.6, 0.9, 1.0) * 600
            x += bw + int(r.uniform(-10, 30) if not far else r.uniform(-20, 10))

    buildings(True)
    buildings(False)
    # dedicated buildings that carry the neon signs
    for txt, x, y, sz, c, nits, fl in SIGNS:
        m, pad = text_mask(txt, sz, True, 0)
        tw, th = m.shape[1] - 2 * pad, m.shape[0] - 2 * pad
        x0, x1, top = x - 40, x + tw + 40, y - 70
        grad = (1 + 3 * smoothstep(top, YB, np.arange(top, YB, dtype=np.float32)))[:, None, None]
        E[top:YB, x0:x1] = col(0.6, 0.55, 0.5) * 0.03 * grad
        for cy in range(y + th + 40, YB - 16, 21):
            for cx in range(x0 + 8, x1 - 12, 15):
                if r.random() < 0.4:
                    E[cy:cy + 12, cx:cx + 8] = col(1.0, 0.72, 0.42) * float(min(np.exp(r.normal(math.log(35), 0.8)), 300))
    # street + lamps
    E[YB:YR] = col(0.5, 0.45, 0.4) * 0.04
    lamps = []
    for lx in range(40, CW, 170):
        E[YB - 40:YB, lx:lx + 2] = 0.01
        E[YB - 42:YB - 38, lx - 3:lx + 6] = col(1.0, 0.62, 0.25) * 2500
        lamps.append(lx)
    Ys, Xs = np.mgrid[YB:YR, 0:CW].astype(np.float32)
    pool = sum(np.exp(-((Xs - lx) / 35) ** 2) for lx in lamps) * np.exp(-(Ys - YB) / 30)
    E[YB:YR] += (pool * 4)[..., None] * col(1.0, 0.62, 0.25)
    # neon signs (patches in canvas coords)
    neon = []
    for txt, x, y, sz, c, nits, fl in SIGNS:
        m, pad = text_mask(txt, sz, True, 0)
        m = np.pad(m, 40)
        core = cv2.GaussianBlur(m, (0, 0), 0.8)
        halo = cv2.GaussianBlur(m, (0, 0), 14)
        neon.append((x - 40 - pad, y - 40 - pad, core, halo, c, nits, fl))
    return E, toggles, beacons, neon


CITY = None


def city_window(off):
    E, toggles, beacons, neon = CITY
    return E[:, off:off + W + 1].copy()


def render_city(tt, fi):
    E, toggles, beacons, neon = CITY
    p = PAN * float(smoothstep(-0.5, 8.0, tt))
    off, frac = int(p), p - int(p)
    win = E[:, off:off + W + 1].copy()
    for (x, y, w, h, c, ph, rate) in toggles:
        if off - w <= x < off + W + 1 and math.sin(tt * rate * 3 + ph) > 0:
            win[y:y + h, x - off:x - off + w] = c
    for (x, y, ph) in beacons:
        if off - 4 <= x < off + W + 1 and ((tt + ph) % 1.3) < 0.18:
            yy0 = max(y - 60, 0)
            dd = np.sqrt((XX[yy0:y + 60] - (x - off)) ** 2 + (YY[yy0:y + 60] - y) ** 2)
            win[yy0:y + 60, :W] += ((1500 * np.exp(-(dd / 2.2) ** 2) + 25 * np.exp(-dd / 25))[..., None]
                                    * col(1.0, 0.08, 0.05))
    for (x, y, core, halo, c, nits, fl) in neon:
        if fl == 1:
            lvl = 0.0 if (int(tt * 12) % 17 in (3, 5, 6)) else 1.0
        elif fl == 2:
            lvl = 0.85 + 0.15 * math.sin(tt * 40)
        else:
            lvl = 1.0
        add_mask(win, core, x - off, y, c * nits * lvl)
        add_mask(win, halo, x - off, y, c * nits * lvl * 0.05)
    # cars along the street
    for i in range(7):
        sp = 120 + 45 * i
        cx = (i * 410 + sp * (tt + 2)) % (CW + 200) - 100 - off
        cy = YB + 22 + (i % 2) * 10
        if -60 < cx < W + 60:
            dir_ = 1 if i % 2 == 0 else -1
            cxx = cx if dir_ == 1 else W - cx
            ys0, ys1, xs0, xs1 = int(cy - 16), int(cy + 16), int(max(cxx - 60, 0)), int(min(cxx + 60, W))
            if xs1 > xs0:
                dx = XX[ys0:ys1, xs0:xs1] - cxx
                dy = YY[ys0:ys1, xs0:xs1] - cy
                head = np.exp(-((dx - 22 * dir_) ** 2 + dy ** 2) / 4) * 3000
                tail = np.exp(-((dx + 22 * dir_) ** 2 + dy ** 2) / 4) * 900
                win[ys0:ys1, xs0:xs1] += head[..., None] * col(1, 0.97, 0.9) + tail[..., None] * col(1, 0.05, 0.03)
    img = win[:, :W] * (1 - frac) + win[:, 1:W + 1] * frac
    # river reflection
    yw = YY[YR:]
    depth = (yw - YR) / (H - YR)
    mapy = (2 * YR - yw - 2).astype(np.float32)
    mapx = (XX[YR:] + (2 + 5 * depth) * np.sin(yw * 1.7 + tt * 5.0)
            + 3 * np.sin(yw * 0.23 - tt * 1.3)).astype(np.float32)
    refl = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    refl = cv2.blur(refl, (3, 9))
    img[YR:] = refl * 0.35 + col(0.2, 0.25, 0.35) * 0.01
    img = bloom(img, 0.06, 5)
    return img


def labels_city(img, tt, a):
    put_text(img, "3  City at night", 80, 70, 44, bold=True, alpha=a)
    put_text(img, "Dim facades, warm windows and searing neon, all in one frame",
             80, 128, 28, nits=170, alpha=a)
    p = PAN * float(smoothstep(-0.5, 8.0, tt))
    ca = a * float(smoothstep(0.8, 1.5, tt))
    hx = 1800 + 60 - p
    callout(img, "Neon tubes ≈ 2,000+ nits", (hx, 0.45 * H - 10), (160, -120), ca)
    callout(img, "Lit windows 5–300 nits", (0.55 * W, 0.70 * H), (-160, 150), a * float(smoothstep(1.8, 2.5, tt)))
    callout(img, "Night sky ≈ 0.01 nits", (0.86 * W, 0.12 * H), (-140, 60), a * float(smoothstep(2.8, 3.5, tt)))
    callout(img, "Water reflects at ~35 %", (0.30 * W, 0.93 * H), (140, -30), a * float(smoothstep(3.6, 4.3, tt)))


# ----------------------------------------------------------------------------
# Scene 4: the ladder
LEVELS = [0.01, 0.1, 1, 10, 100, 400, 1000, 4000]


def fmt_nits(v):
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:.0f}"
    return f"{v:g}"


def render_ladder(u, fi):
    img = np.zeros((H, W, 3), np.float32)
    sz, gap, x0, y0 = 170, 50, 105, 250
    for i, L in enumerate(LEVELS):
        on = float(smoothstep(0.3 + i * 0.32, 0.6 + i * 0.32, u))
        lvl = pq_decode(pq_encode(L) * on) if on > 0 else 0.0
        x = x0 + i * (sz + gap)
        img[y0:y0 + sz, x:x + sz] = lvl
    # PQ-uniform gradient bar with wipe-in
    gx0, gx1, gy0, gy1 = 105, 1815, 690, 780
    wipe = float(smoothstep(3.0, 4.2, u))
    xs = np.arange(gx0, gx1, dtype=np.float32)
    e = (xs - gx0) / (gx1 - gx0) * pq_encode(PEAK)
    ramp = pq_decode(e) * (xs < gx0 + wipe * (gx1 - gx0) + 1)
    img[gy0:gy1, gx0:gx1] = ramp[None, :, None]
    return img


def labels_ladder(img, u, a):
    sz, gap, x0, y0 = 170, 50, 105, 250
    put_text(img, "How bright is bright?", W // 2, 70, 56, bold=True, alpha=a, anchor="c")
    put_text(img, "Each step is 10× brighter (100 nits is SDR white, 4,000 is this video's peak)",
             W // 2, 150, 28, nits=170, alpha=a, anchor="c")
    for i, L in enumerate(LEVELS):
        la = a * float(smoothstep(0.3 + i * 0.32, 0.6 + i * 0.32, u))
        put_text(img, fmt_nits(L) + (" nits" if i == 0 else ""), x0 + i * (sz + gap) + sz // 2, y0 + sz + 22, 30,
                 nits=150, alpha=la, anchor="c")
    ba = a * float(smoothstep(2.8, 3.3, u))
    # brackets: SDR vs HDR headroom
    m = np.zeros((H, W), np.float32)
    yb = y0 + sz + 78
    cv2.line(m, (x0, yb), (x0 + 4 * (sz + gap) + sz, yb), 1.0, 2, cv2.LINE_AA)
    m2 = np.zeros((H, W), np.float32)
    cv2.line(m2, (x0 + 5 * (sz + gap), yb), (x0 + 7 * (sz + gap) + sz, yb), 1.0, 2, cv2.LINE_AA)
    img += (m * 120 * ba)[..., None] + (m2 * 120 * ba)[..., None] * col(1, 0.75, 0.35)
    put_text(img, "SDR range", x0 + 2 * (sz + gap) + sz // 2, yb + 14, 28, nits=120, alpha=ba, anchor="c")
    put_text(img, "HDR headroom: 5+ more stops", x0 + 6 * (sz + gap) + sz // 2, yb + 14, 28, nits=150,
             rgb=(1, 0.8, 0.5), alpha=ba, anchor="c")
    ga = a * float(smoothstep(3.6, 4.4, u))
    gx0, gx1, gy1 = 105, 1815, 780
    for L in [0.01, 0.1, 1, 10, 100, 1000, 4000]:
        x = gx0 + float(pq_encode(L) / pq_encode(PEAK)) * (gx1 - gx0)
        tick = np.zeros((H, W), np.float32)
        cv2.line(tick, (int(x), gy1 + 4), (int(x), gy1 + 18), 1.0, 2, cv2.LINE_AA)
        img += (tick * 140 * ga)[..., None]
        put_text(img, fmt_nits(L), x, gy1 + 30, 26, nits=140, alpha=ga, anchor="c")
    put_text(img, "Smooth 10-bit PQ ramp, 0 → 4,000 nits", W // 2, 640, 28, nits=150, alpha=ga, anchor="c")
    fa = a * float(smoothstep(4.8, 5.4, u))
    put_text(img, "HDR10  ·  SMPTE ST 2084 (PQ)  ·  BT.2020  ·  mastered to 4,000 nits", W // 2, 920, 30,
             nits=TEXT_NITS, alpha=fa, anchor="c")


# ----------------------------------------------------------------------------
def readout(img, a):
    """Measure the frame (before overlays) and print a live dynamic-range readout."""
    lum = img @ col(0.2126, 0.7152, 0.0722)
    small = cv2.resize(lum, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    peak = float(min(lum.max(), PEAK))
    floor = float(max(np.percentile(small, 0.5), 0.0005))
    peak = max(peak, floor)
    stops = math.log2(peak / floor)
    s = f"Peak {peak:,.0f} nits   ·   Darkest {floor:.3f} nits   ·   {stops:.1f} stops"
    put_text(img, s, 80, H - 70, 28, nits=180, alpha=a)
    put_text(img, "HDR10 · PQ · BT.2020", W - 80, H - 70, 26, nits=140, alpha=a * 0.8, anchor="r")


SCENES = [(S1, S2, render_sunset, labels_sunset), (S2, S3, render_fireworks, labels_fireworks),
          (S3, S4, render_city, labels_city), (S4, DUR, render_ladder, labels_ladder)]


def render_frame(fi):
    t = fi / FPS
    for i, (a, b, rend, lab) in enumerate(SCENES):
        if a <= t < b or (i == len(SCENES) - 1 and t >= b):
            tt = t - a
            img = rend(tt, fi)
            # dip between scenes (applied in linear light with a perceptual-ish curve)
            fin = 0.45 if i else 0.8
            fout = 0.8 if i == len(SCENES) - 1 else 0.45
            env = float(smoothstep(a, a + fin, t) * (1 - smoothstep(b - fout, b, t)))
            img *= env ** 2.4
            la = scene_alpha(t, a + 0.2, b, 0.5)
            if i < 3:
                readout(img, la)
            lab(img, tt, la)
            return np.clip(img, 0, None)


def to_2020(img709):
    return np.clip(img709 @ M709_2020.T, 0, PEAK)


def frame_stats(rgb2020):
    mx = rgb2020.max(axis=2)
    return float(mx.max()), float(mx.mean())


def to_yuv420p10(rgb2020):
    e = pq_encode(rgb2020).astype(np.float32)
    R, G, B = e[..., 0], e[..., 1], e[..., 2]
    Y = 0.2627 * R + 0.6780 * G + 0.0593 * B
    Cb = (B - Y) / 1.8814
    Cr = (R - Y) / 1.4746
    Cb = cv2.resize(Cb, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    Cr = cv2.resize(Cr, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    y10 = np.clip(np.round(64 + 876 * Y), 64, 940).astype(np.uint16)
    cb10 = np.clip(np.round(512 + 896 * Cb), 64, 960).astype(np.uint16)
    cr10 = np.clip(np.round(512 + 896 * Cr), 64, 960).astype(np.uint16)
    return y10.tobytes() + cb10.tobytes() + cr10.tobytes()


def sdr_preview(img709):
    """Log-compressed view for eyeballing on an SDR screen (not the real look)."""
    v = np.log10(1 + img709 * 10) / math.log10(1 + PEAK * 10)
    return (np.clip(v, 0, 1) * 255).astype(np.uint8)[..., ::-1]


def work_stats(fi):
    return fi, frame_stats(to_2020(render_frame(fi)))


def work_encode(fi):
    rgb = to_2020(render_frame(fi))
    return to_yuv420p10(rgb)


# ----------------------------------------------------------------------------
# audio
def make_audio(path):
    sr = 48000
    n = int(DUR * sr)
    t = np.arange(n) / sr
    out = np.zeros((n, 2), np.float64)
    rng = np.random.default_rng(1)

    def env(a, b, att, rel):
        return smoothstep(a, a + att, t) * (1 - smoothstep(b - rel, b, t))

    def pad(freqs, a, b, amp, att=1.5, rel=1.5):
        e = env(a, b, att, rel)
        for k, f in enumerate(freqs):
            for det, pan in ((-0.15, 0.3), (0.15, 0.7)):
                ph = rng.uniform(0, 6.28)
                s = np.sin(2 * np.pi * (f + det) * t + ph + 0.3 * np.sin(2 * np.pi * 0.2 * t + k))
                s += 0.25 * np.sin(2 * np.pi * 2 * (f + det) * t + ph)
                out[:, 0] += s * e * amp * (1 - pan)
                out[:, 1] += s * e * amp * pan

    def note(f, at, amp, dec=1.2, pan=0.5):
        i0 = int(at * sr)
        m = min(n - i0, int(dec * 5 * sr))
        if m <= 0:
            return
        tt = np.arange(m) / sr
        s = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 2.01 * f * tt) * np.exp(-tt * 3)) \
            * np.exp(-tt / dec) * np.minimum(tt / 0.005, 1)
        out[i0:i0 + m, 0] += s * amp * (1 - pan)
        out[i0:i0 + m, 1] += s * amp * pan

    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    # 1: warm Dmaj9 pad
    pad([hz(m) for m in (50, 57, 61, 64, 66)], 0.0, 8.6, 0.05, 2.0, 1.2)
    # 2: low drone + booms
    pad([hz(m) for m in (38, 45)], 7.6, 16.5, 0.06, 1.0, 1.0)
    for i, b in enumerate(BURSTS):
        t0 = S2 + b["t0"]
        pan = b["x"] / W
        # launch whistle
        i0, m = int((t0 - 0.9) * sr), int(0.9 * sr)
        if i0 > int(S2 * sr):
            tt = np.arange(m) / sr
            f = 900 + 1400 * tt / 0.9
            s = np.sin(2 * np.pi * np.cumsum(f) / sr) * 0.03 * (tt / 0.9)
            out[i0:i0 + m, 0] += s * (1 - pan)
            out[i0:i0 + m, 1] += s * pan
        # boom (slightly delayed: sound is slower than light)
        i0 = int((t0 + 0.2) * sr)
        m = min(n - i0, int(2.5 * sr))
        tt = np.arange(m) / sr
        f = 40 + 60 * np.exp(-tt * 8)
        boom = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-tt * 2.5) * 0.5
        nz = rng.normal(size=m) * np.exp(-tt * 9) * 0.12
        nz = np.convolve(nz, np.ones(6) / 6, "same")
        s = boom + nz
        if b["crackle"]:
            for _ in range(140):
                k = int(rng.uniform(0.9, 2.4) * sr)
                if k < m - 200:
                    s[k:k + 120] += rng.normal(size=120) * np.exp(-np.arange(120) / 25) * 0.25
        out[i0:i0 + m, 0] += s * (1.2 - pan)
        out[i0:i0 + m, 1] += s * (0.2 + pan)
    # 3: city pulse + arpeggio
    pad([hz(m) for m in (45, 52, 57, 60)], 15.6, 24.0, 0.04, 1.2, 1.0)
    scale = [69, 72, 74, 76, 79, 81, 84]
    for k in range(int((S4 - S3 - 0.5) / 0.25)):
        note(hz(scale[(k * 3 + k // 4) % len(scale)]), S3 + 0.2 + k * 0.25, 0.05, 0.35, 0.3 + 0.4 * ((k * 5) % 7) / 6)
    # 4: rising notes for each patch, final chord
    for i in range(len(LEVELS)):
        note(hz(62 + [0, 2, 4, 7, 9, 12, 14, 16][i]), S4 + 0.3 + i * 0.32, 0.12, 1.4, 0.2 + 0.08 * i)
    pad([hz(m) for m in (50, 57, 62, 66, 69, 74)], S4 + 2.8, DUR, 0.05, 1.5, 1.8)
    # reverb
    ir_t = np.arange(int(1.8 * sr)) / sr
    for c in range(2):
        ir = rng.normal(size=ir_t.size) * np.exp(-ir_t / 0.5)
        ir[0] = 0
        wet = fftconvolve(out[:, c], ir)[:n] * 0.02
        out[:, c] = out[:, c] + wet
    out[-int(0.05 * sr):] *= np.linspace(1, 0, int(0.05 * sr))[:, None]
    out /= np.abs(out).max() / 0.89
    wavfile.write(path, sr, out.astype(np.float32))


# ----------------------------------------------------------------------------
def main():
    global CITY
    CITY = build_city()
    mode = sys.argv[1] if len(sys.argv) > 1 else "stats"
    stats_path = os.path.join(SCRATCH, "hdr_stats.json")
    nproc = max(1, (os.cpu_count() or 4) - 2)
    if mode == "still":
        for s in sys.argv[2:]:
            fi = int(float(s) * FPS)
            img = render_frame(fi)
            cv2.imwrite(os.path.join(SCRATCH, f"preview_{fi:04d}.png"), sdr_preview(img))
            print(fi, "peak", img.max())
        return
    if mode == "stats":
        with Pool(nproc) as p:
            res = dict(p.imap_unordered(work_stats, range(NFRAMES), chunksize=4))
        maxcll = max(v[0] for v in res.values())
        maxfall = max(v[1] for v in res.values())
        json.dump({"maxcll": maxcll, "maxfall": maxfall}, open(stats_path, "w"))
        print(f"MaxCLL {maxcll:.0f}  MaxFALL {maxfall:.1f}")
        return
    if mode == "encode":
        st = json.load(open(stats_path))
        cll, fall = int(math.ceil(st["maxcll"])), int(math.ceil(st["maxfall"]))
        wav = os.path.join(SCRATCH, "hdr_audio.wav")
        make_audio(wav)
        out = os.path.join(HERE, "hdr_demo.mp4")
        x265 = (f"hdr10=1:hdr10-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:"
                f"colormatrix=bt2020nc:range=limited:chromaloc=0:"
                f"master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(40000000,50):"
                f"max-cll={cll},{fall}:keyint=60:bframes=4")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-stats",
               "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", f"{W}x{H}", "-r", str(FPS),
               "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
               "-color_range", "tv", "-i", "-", "-i", wav,
               "-c:v", "libx265", "-preset", "slow", "-b:v", "12M", "-maxrate", "20M", "-bufsize", "24M",
               "-pix_fmt", "yuv420p10le",
               "-x265-params", x265, "-tag:v", "hvc1",
               "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
               "-color_range", "tv",
               "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart", "-shortest", out]
        ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        with Pool(nproc) as p:
            for i, buf in enumerate(p.imap(work_encode, range(NFRAMES), chunksize=2)):
                ff.stdin.write(buf)
        ff.stdin.close()
        ff.wait()
        print("wrote", out, "MaxCLL", cll, "MaxFALL", fall)


if __name__ == "__main__":
    main()
