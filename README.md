# HDR Demo Video

A 30-second, 1920×1080 **HDR10** video for YouTube, generated entirely in code.
Its purpose is to show how much dynamic range a modern HDR display can deliver.

Every frame is rendered in linear light, in absolute units (nits, cd/m²).
Each frame is then converted to BT.2020, encoded with the SMPTE ST 2084 (PQ) curve and turned into 10-bit limited-range Y′CbCr 4:2:0.
Finally the frames are piped straight into x265 with HDR10 static metadata.

## Contents

| Time | Scene | What it shows |
|------|-------|---------------|
| 0–8 s | Sunset over water | Sun disk ≈ 4,000 nits, glitter on the water up to 3,600 nits, shadowed headland ≈ 0.03–0.1 nits |
| 8–16 s | Fireworks at night | Sparks up to 4,000 nits against a 0.003-nit starry sky, with hills lit by the flashes |
| 16–23.5 s | City at night | Neon ≈ 2,000+ nits, windows 5–300 nits, a 0.01-nit sky, moving car lights and river reflections |
| 23.5–30 s | "How bright is bright?" | Patches stepping 10× each from 0.01 to 4,000 nits, the SDR range vs. the HDR headroom, and a smooth 10-bit PQ ramp |

Scenes 1–3 carry a live readout of the measured peak brightness, the darkest level (0.5th percentile) and the range in stops.
Text is drawn at 203 nits, the BT.2408 reference level for graphics white.
The soundtrack is synthesized in the same script and is synchronized to the visuals, for example the firework launches and booms.

## Output format

- HEVC Main 10, `yuv420p10le`, 30 fps, `hvc1` tag, MP4 with faststart
- BT.2020 primaries, PQ transfer, BT.2020 non-constant-luminance matrix, limited range
- Mastering display: P3-D65 primaries, 0.005–4,000 nits
- MaxCLL / MaxFALL are measured from the rendered frames (currently 4000 / 116)
- AAC stereo at 48 kHz and 320 kb/s

## Requirements

- Python 3 with `numpy`, `scipy`, `opencv-python` (`cv2`) and `Pillow`
- `ffmpeg` built with `libx265`
- DejaVu fonts (`/usr/share/fonts/truetype/dejavu/`)

## Usage

```sh
python3 hdr_demo.py stats      # pass 1: measure MaxCLL/MaxFALL -> hdr_stats.json
python3 hdr_demo.py encode     # pass 2: render, synthesize audio, encode hdr_demo.mp4
python3 hdr_demo.py still 3 12.5 27   # write log-compressed SDR preview PNGs at those times
```

Rendering is parallelized across all CPU cores but two.
On a 28-core machine the stats pass takes about 75 s and the encode about 2.5 min.

Intermediate files (`hdr_stats.json`, `hdr_audio.wav` and the preview PNGs) go to the script's directory.
To send them somewhere else, set `HDR_SCRATCH=/some/dir`.

## Notes

- The preview PNGs use a log curve so that the whole range fits on an SDR screen. They do **not** show what the video looks like on an HDR display.
- The encode uses `-crf 14`, which averages about 4 Mb/s. YouTube recommends about 10 Mb/s for 1080p HDR uploads, so lower the CRF if you want more headroom for the dark gradients.
- YouTube can take a while after an upload finishes before it offers the HDR version.
