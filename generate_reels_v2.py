#!/usr/bin/env python3
"""
YTP Reels Generator v2
Reads routes from data_enhance/enrichment.json.
Features: Ken Burns zoom, slide transitions, caption overlays,
          smart photo selection, emoji stripping,
          beat-synced transitions (librosa), AI voiceover (edge-tts).
"""

import os
import re
import json
import glob
import asyncio
import tempfile
import numpy as np
import librosa
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    VideoClip, ImageClip, AudioFileClip,
    concatenate_videoclips, CompositeVideoClip, CompositeAudioClip,
)
from moviepy import vfx, afx
import edge_tts

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ENHANCE_DIR = os.path.join(BASE_DIR, "data_enhance")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
BGM_PATH    = os.path.join(BASE_DIR, "bgm2.mp3")
JSON_PATH   = os.path.join(ENHANCE_DIR, "enrichment.json")

# ── Video settings ───────────────────────────────────────────────────────────
W, H = 1080, 1920   # 9:16 vertical Reels format
FPS  = 30
PHOTOS_PER_STOP = 3
PHOTO_DUR  = 2.8    # seconds per photo
TRANS_DUR  = 0.45   # transition duration (fade / slide)
ZOOM_START = 1.0
ZOOM_END   = 1.10   # Ken Burns zoom range

# ── Fonts (macOS) ────────────────────────────────────────────────────────────
FONT_PATH = "/System/Library/Fonts/STHeiti Medium.ttc"

# ── Emoji strip ──────────────────────────────────────────────────────────────
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"   # misc symbols, emoticons, transport, map
    "\U0001FA00-\U0001FAFF"   # extended symbols
    "\u2600-\u26FF"           # misc symbols (✈ ☕ ☀)
    "\u2700-\u27BF"           # dingbats (✨ ✅)
    "\uFE00-\uFE0F"           # variation selectors (️ suffix)
    "\u200d"                  # zero-width joiner (used in ZWJ emoji sequences)
    "\u200b"                  # zero-width space
    "]+",
    flags=re.UNICODE,
)

def strip_emoji(text: str) -> str:
    """Remove emoji and invisible joiners that STHeiti cannot render."""
    return _EMOJI_RE.sub("", text).strip()


# ── Beat sync ────────────────────────────────────────────────────────────────

BEATS_PER_PHOTO = 6          # how many beats each photo lasts
TTS_VOICE       = "zh-TW-HsiaoChenNeural"   # Traditional Chinese female

def analyze_bgm(bgm_path: str, beats_per_photo: int = BEATS_PER_PHOTO) -> tuple[float, float, float]:
    """
    Analyse the BGM and return (bgm_start, photo_dur, beat_interval) where:
      bgm_start     — timestamp (s) of the first detected beat (trim silence)
      photo_dur     — duration (s) of one photo = beats_per_photo × beat interval
      beat_interval — average duration (s) of one beat
    """
    y, sr = librosa.load(bgm_path)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times     = librosa.frames_to_time(beat_frames, sr=sr)
    avg_interval   = float(np.mean(np.diff(beat_times)))
    bgm_start      = float(beat_times[0]) if len(beat_times) else 0.0
    photo_dur      = round(avg_interval * beats_per_photo, 3)
    print(f"  ♪ BGM: {60/avg_interval:.1f} BPM  |  photo_dur = {photo_dur}s ({beats_per_photo} beats)")
    return bgm_start, photo_dur, avg_interval


# ── AI Voiceover (edge-tts) ───────────────────────────────────────────────────

async def _tts_save(text: str, path: str) -> None:
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    await communicate.save(path)

def tts_script(place: dict) -> str:
    """Build a short narration line for one stop."""
    name    = strip_emoji(place.get("place_name", ""))
    caps    = place.get("captions", [])
    caption = strip_emoji(caps[0]) if caps else ""
    return f"{name}。{caption}。"

def generate_tts(route: dict, tmp_dir: str) -> tuple[str, list[str]]:
    """
    Generate TTS mp3 files: one intro + one per stop.
    Returns (intro_path, [stop_0_path, stop_1_path, ...]).
    """
    intro_text = f"今天帶你探索{route['route_name']}，出發！"
    intro_path = os.path.join(tmp_dir, "tts_intro.mp3")
    asyncio.run(_tts_save(intro_text, intro_path))
    print(f"  🎙 TTS intro: {intro_text}")

    stop_paths = []
    for i, place in enumerate(route["places"]):
        script = tts_script(place)
        path   = os.path.join(tmp_dir, f"tts_stop_{i}.mp3")
        asyncio.run(_tts_save(script, path))
        stop_paths.append(path)
        print(f"  🎙 TTS stop {i}: {script[:30]}…")
    return intro_path, stop_paths


# ── Plog warm filter ─────────────────────────────────────────────────────────

def apply_plog_filter(arr: np.ndarray) -> np.ndarray:
    """
    Film-diary / Plog look:
      • Warm colour grade  — lift reds, cool down blues
      • Vignette           — darken edges like an old lens
      • Subtle film grain  — adds texture / analogue feel
    Applied to the photo BEFORE the text overlay so text stays crisp.
    """
    f = arr.astype(np.float32)

    # Warm grade
    f[:, :, 0] = np.clip(f[:, :, 0] * 1.08 + 6, 0, 255)   # red up
    f[:, :, 1] = np.clip(f[:, :, 1] * 1.02,      0, 255)   # green slight
    f[:, :, 2] = np.clip(f[:, :, 2] * 0.88,      0, 255)   # blue down (warm)

    # Vignette — smooth radial darkening from the edges
    h, w = arr.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    # clamp: centre stays at 1.0, far corners reach ~0.55
    vignette = np.clip(1.0 - dist * 0.38, 0.55, 1.0)
    f *= vignette[:, :, np.newaxis]

    # Grain — very subtle (σ=6), only add, don't remove (keeps highlights bright)
    rng = np.random.default_rng(seed=42)          # deterministic per-frame
    grain = rng.normal(0, 6, arr.shape).astype(np.float32)
    f = np.clip(f + grain, 0, 255)

    return f.astype(np.uint8)


# ── Photo selection ──────────────────────────────────────────────────────────

def sharpness_score(img_path: str) -> float:
    """
    Laplacian-variance sharpness score (higher = sharper / more detail).
    Works with PIL + numpy only, no OpenCV needed.
    """
    try:
        arr = np.array(Image.open(img_path).convert("L")).astype(np.float32)
        # 2nd-order finite differences along each axis
        gy = arr[2:] - 2 * arr[1:-1] + arr[:-2]
        gx = arr[:, 2:] - 2 * arr[:, 1:-1] + arr[:, :-2]
        return float(np.var(gy)) + float(np.var(gx))
    except Exception:
        return 0.0


def pick_best_photos(folder: str, filenames: list[str], n: int = PHOTOS_PER_STOP) -> list[str]:
    """
    Score every photo by sharpness and return the top-n paths, preserving
    their original numeric order so the reel flows naturally.
    """
    scored = []
    for fn in filenames:
        path = os.path.join(folder, fn)
        if os.path.exists(path):
            scored.append((sharpness_score(path), fn, path))

    if not scored:
        return []

    # Pick top-n by score, then re-sort by original filename order
    top = sorted(scored, key=lambda x: -x[0])[:n]
    top_fns = {fn for _, fn, _ in top}
    ordered = [path for _, fn, path in scored if fn in top_fns]
    return ordered[:n]


# ── Image helpers ────────────────────────────────────────────────────────────

def crop_to_9_16(pil_img: Image.Image) -> Image.Image:
    """Center-crop and resize to W×H (1080×1920)."""
    iw, ih = pil_img.size
    if iw / ih > W / H:
        new_w = int(ih * W / H)
        left  = (iw - new_w) // 2
        pil_img = pil_img.crop((left, 0, left + new_w, ih))
    else:
        new_h = int(iw * H / W)
        top   = (ih - new_h) // 2
        pil_img = pil_img.crop((0, top, iw, top + new_h))
    return pil_img.resize((W, H), Image.LANCZOS)


def add_text_overlay(
    pil_img: Image.Image,
    name_str: str,
    caption_str: str,
) -> np.ndarray:
    """
    Composite a gradient bar + place name + caption onto the image.
    Layout (bottom section):
      [name]    — large white
      [caption] — smaller warm-yellow
    """
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    bar_start = H - 340
    for y in range(bar_start, H):
        alpha = int(220 * (y - bar_start) / (H - bar_start))
        draw_ov.line([(0, y), (W, y)], fill=(0, 0, 15, alpha))

    base = pil_img.convert("RGBA")
    base = Image.alpha_composite(base, overlay).convert("RGB")
    draw = ImageDraw.Draw(base)

    try:
        f_name    = ImageFont.truetype(FONT_PATH, 72, index=0)
        f_caption = ImageFont.truetype(FONT_PATH, 46, index=0)
    except Exception:
        f_name    = ImageFont.load_default()
        f_caption = ImageFont.load_default()

    # Place name (white, bold-ish)
    draw.text((70, bar_start + 30), strip_emoji(name_str), font=f_name, fill=(255, 255, 255))
    # Caption (warm golden) — strip emoji so STHeiti doesn't render □
    draw.text((70, bar_start + 140), strip_emoji(caption_str), font=f_caption, fill=(255, 210, 50))

    return np.array(base)


# ── Ken Burns clip ────────────────────────────────────────────────────────────

def make_ken_burns_clip(
    frame: np.ndarray,
    duration: float,
    zoom_start: float = ZOOM_START,
    zoom_end: float = ZOOM_END,
) -> VideoClip:
    """
    Wrap a static frame in a VideoClip that slowly zooms in (Ken Burns).
    The zoom is purely a centre-crop resize — no quality loss on the final render.
    """
    h, w = frame.shape[:2]

    def make_frame(t: float) -> np.ndarray:
        progress = t / duration
        zoom = zoom_start + (zoom_end - zoom_start) * progress
        # Crop a smaller window and upscale to W×H
        crop_w = int(w / zoom)
        crop_h = int(h / zoom)
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        cropped = frame[y1: y1 + crop_h, x1: x1 + crop_w]
        img = Image.fromarray(cropped).resize((w, h), Image.LANCZOS)
        return np.array(img)

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


# ── Slide transition helper ───────────────────────────────────────────────────

def slide_transition(clip_a: VideoClip, clip_b: VideoClip, direction: str = "left") -> VideoClip:
    """
    Slide clip_b in over clip_a during the last TRANS_DUR seconds.
    direction: 'left' → clip_b enters from the right edge
               'right' → clip_b enters from the left edge
    Returns a composite clip of duration TRANS_DUR.
    """
    # Trim both clips to the transition window
    seg_a = clip_a.subclipped(clip_a.duration - TRANS_DUR, clip_a.duration)
    seg_b = clip_b.subclipped(0, TRANS_DUR)

    if direction == "left":
        # clip_b slides from right (x goes W→0)
        def pos_b(t):
            progress = t / TRANS_DUR
            x = int(W * (1 - progress))
            return (x, 0)
    else:
        # clip_b slides from left (x goes -W→0)
        def pos_b(t):
            progress = t / TRANS_DUR
            x = int(-W * (1 - progress))
            return (x, 0)

    seg_b_moving = seg_b.with_position(pos_b)
    composite = CompositeVideoClip([seg_a, seg_b_moving], size=(W, H))
    composite = composite.with_duration(TRANS_DUR)
    return composite


# ── Clip builders ─────────────────────────────────────────────────────────────

def make_stop_clips(stop: dict, route_folder: str, photo_dur: float = PHOTO_DUR,
                    first_photo_dur: float | None = None) -> list[VideoClip]:
    """Build Ken-Burns ImageClips for one itinerary stop.
    first_photo_dur: if set, first photo uses this duration (holds until TTS ends).
    """
    folder   = os.path.join(ENHANCE_DIR, route_folder, stop["folder"])
    photos   = stop.get("photos", [])
    captions = stop.get("captions", [])
    name     = stop.get("place_name", stop.get("name", ""))

    best_paths = pick_best_photos(folder, photos, PHOTOS_PER_STOP)

    if not best_paths:
        print(f"  ⚠ No photos found in: {folder}")
        blank = np.zeros((H, W, 3), dtype=np.uint8)
        return [make_ken_burns_clip(blank, photo_dur)]

    clips = []
    for i, path in enumerate(best_paths):
        caption = captions[i] if i < len(captions) else ""
        pil_img = crop_to_9_16(Image.open(path).convert("RGB"))
        frame   = add_text_overlay(pil_img, name, caption)

        # Alternate zoom direction: even → zoom in, odd → zoom out
        z_start, z_end = (ZOOM_START, ZOOM_END) if i % 2 == 0 else (ZOOM_END, ZOOM_START)
        dur = (first_photo_dur if i == 0 and first_photo_dur else photo_dur)
        clip = make_ken_burns_clip(frame, dur, z_start, z_end)
        clips.append(clip)
    return clips


# ── Route builder ─────────────────────────────────────────────────────────────

def make_route_video(route_id: str, route: dict) -> None:
    route_folder = route_id  # e.g. "route_A"
    print(f"\n── {route_id}: {route['route_name']} ──")

    # ── Beat sync: derive photo_dur from the BGM ─────────────────────────────
    if os.path.exists(BGM_PATH):
        bgm_start, photo_dur, beat_interval = analyze_bgm(BGM_PATH)
    else:
        bgm_start, photo_dur, beat_interval = 0.0, PHOTO_DUR, PHOTO_DUR / BEATS_PER_PHOTO
        print("  (no BGM — using default photo duration)")

    # ── AI voiceover ─────────────────────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="ytp_tts_")
    print("  Generating TTS voiceover…")
    intro_path, stop_paths = generate_tts(route, tmp_dir)

    # Intro duration — stop 0's TTS starts after intro finishes
    intro_dur = AudioFileClip(intro_path).duration if os.path.exists(intro_path) else 0.0

    stop_clip_groups: list[list[VideoClip]] = []
    for stop in route["places"]:
        print(f"  {stop['place_name']}")
        clips = make_stop_clips(stop, route_folder, photo_dur)
        stop_clip_groups.append(clips)

    # Build final segment list:
    #   • very first clip → FadeIn from black (opening)
    #   • within same stop → clean cut  (Ken Burns already gives motion)
    #   • between stops   → slide transition (left/right alternating)
    final_segments: list[VideoClip] = []
    direction_toggle = 0

    for g_idx, group in enumerate(stop_clip_groups):
        for c_idx, clip in enumerate(group):
            is_first_clip = (g_idx == 0 and c_idx == 0)
            is_first_of_stop = (c_idx == 0)

            if is_first_clip:
                # Opening: fade in from black once
                final_segments.append(clip.with_effects([vfx.FadeIn(TRANS_DUR)]))

            elif is_first_of_stop:
                # New stop: slide the incoming clip over the last outgoing clip
                direction = "left" if direction_toggle % 2 == 0 else "right"
                direction_toggle += 1
                prev = final_segments[-1]
                trans = slide_transition(prev, clip, direction)
                final_segments[-1] = prev.subclipped(0, prev.duration - TRANS_DUR)
                final_segments.append(trans)
                remainder = clip.subclipped(TRANS_DUR, clip.duration)
                if remainder.duration > 0.05:
                    final_segments.append(remainder)

            else:
                # Within same stop: clean cut — no fade, no effect
                final_segments.append(clip)

    final = concatenate_videoclips(final_segments, method="compose")

    # ── Audio: BGM + TTS voiceover ────────────────────────────────────────────
    audio_layers = []

    if os.path.exists(BGM_PATH):
        bgm = (
            AudioFileClip(BGM_PATH)
            .subclipped(bgm_start, bgm_start + final.duration)
            .with_effects([afx.AudioFadeIn(1.0), afx.AudioFadeOut(2.0)])
            .with_volume_scaled(0.55)       # duck BGM so TTS is clear
        )
        audio_layers.append(bgm)

    # TTS start times: each stop's TTS starts exactly when that stop's first photo appears
    # stop_i start = sum of previous stops' total durations
    # total duration of stop_i = first_photo_dur_i + (PHOTOS_PER_STOP-1) * photo_dur
    # TTS start times with no-overlap guarantee:
    # intro → t=0
    # stop i → max(stop_natural_start, prev_tts_end)
    stop_dur = PHOTOS_PER_STOP * photo_dur   # natural duration of one stop
    tts_items = [(intro_path, 0.0)]
    prev_tts_end = intro_dur   # after intro, stop 0 TTS can begin
    for i, tts_path in enumerate(stop_paths):
        stop_natural = i * stop_dur
        t_start = max(stop_natural, prev_tts_end)
        tts_dur = AudioFileClip(tts_path).duration if os.path.exists(tts_path) else 0.0
        prev_tts_end = t_start + tts_dur
        if i > 0 and t_start > stop_natural:
            print(f"  ⚠ TTS stop {i} pushed to {t_start:.2f}s (overlap guard)")
        tts_items.append((tts_path, t_start))

    for tts_path, t_start in tts_items:
        if os.path.exists(tts_path) and t_start < final.duration:
            tts_clip = (
                AudioFileClip(tts_path)
                .with_start(t_start)
                .with_volume_scaled(1.0)
            )
            audio_layers.append(tts_clip)

    if audio_layers:
        final = final.with_audio(CompositeAudioClip(audio_layers))
    else:
        print("  (no audio)")

    safe_name = route["route_name"].replace("/", "_").replace(" ", "_")
    out_name  = f"{route_id}_{safe_name}.mp4"
    out_path  = os.path.join(OUTPUT_DIR, out_name)

    final.write_videofile(
        out_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="fast",
        logger="bar",
    )
    print(f"  ✓ Saved → {out_path}")
    final.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(JSON_PATH, encoding="utf-8") as f:
        enrichment = json.load(f)

    routes = enrichment["routes"]
    for route_id, route in routes.items():
        make_route_video(route_id, route)

    print("\nDone! All videos saved to:", OUTPUT_DIR)
