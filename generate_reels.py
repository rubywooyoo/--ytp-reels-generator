#!/usr/bin/env python3
"""
YTP Reels Generator
Generates 3 travel recommendation Reels (1080x1920, 9:16) from local photos.
"""

import os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips, ColorClip
from moviepy import vfx, afx

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
BGM_PATH   = os.path.join(BASE_DIR, "bgm.mp3")  # optional

# ── Video settings ───────────────────────────────────────────────────────────
W, H = 1080, 1920   # 9:16 vertical Reels format
FPS  = 30
PHOTOS_PER_STOP = 3
PHOTO_DUR  = 2.5    # seconds per photo
FADE_DUR   = 0.4    # fade in/out between clips

# ── Fonts (macOS) ────────────────────────────────────────────────────────────
FONT_PATH = "/System/Library/Fonts/STHeiti Medium.ttc"

# ── Three route definitions ──────────────────────────────────────────────────
ROUTES = [
    {
        "title":  "路線一｜文青探索",
        "output": "route_1_indie.mp4",
        "stops": [
            {"time": "09:30", "name": "臺北市立美術館",
             "folder": "臺北市立美術館_place/1_臺北市立美術館"},
            {"time": "12:00", "name": "稻舍食館 迪化",
             "folder": "餐廳_results/7_稻舍食館 迪化店"},
            {"time": "14:00", "name": "臺北玫瑰園",
             "folder": "臺北玫瑰園_place/1_臺北玫瑰園"},
            {"time": "16:00", "name": "吟遊壺裡 Coffee",
             "folder": "咖啡廳_results/6_吟遊壺裡 Coffee Roasters"},
            {"time": "19:00", "name": "布娜飛紅酒館",
             "folder": "餐酒_results/6_布娜飛紅酒餐酒館Bravo Wine"},
        ],
    },
    {
        "title":  "路線二｜輕鬆愜意",
        "output": "route_2_chill.mp4",
        "stops": [
            {"time": "09:30", "name": "臺北市立美術館",
             "folder": "臺北市立美術館_place/1_臺北市立美術館"},
            {"time": "12:00", "name": "TakeOut Burger & Cafe",
             "folder": "餐廳_results/6_TakeOut Burger&Cafe 民權店"},
            {"time": "14:00", "name": "臺北玫瑰園",
             "folder": "臺北玫瑰園_place/1_臺北玫瑰園"},
            {"time": "16:00", "name": "Oasis Coffee & Bar",
             "folder": "咖啡廳_results/3_Oasis Coffee & Bar"},
            {"time": "19:00", "name": "歐吧噠韓餐酒 圓山",
             "folder": "餐酒_results/1_歐吧噠韓餐酒 圓山花博店 오빠닭 감성포차 Oppadak Korean Cuisine"},
        ],
    },
    {
        "title":  "路線三｜精緻海鮮",
        "output": "route_3_seafood.mp4",
        "stops": [
            {"time": "09:30", "name": "臺北市立美術館",
             "folder": "臺北市立美術館_place/1_臺北市立美術館"},
            {"time": "12:00", "name": "安安生魚小料理",
             "folder": "餐廳_results/9_安安生魚小料理（位子很少請於營業時間致電訂位💕）"},
            {"time": "14:00", "name": "臺北玫瑰園",
             "folder": "臺北玫瑰園_place/1_臺北玫瑰園"},
            {"time": "16:00", "name": "What's Life Coffee",
             "folder": "咖啡廳_results/4_What's Life Coffee Roasters（最多四位入內及保持低聲細語勿大聲交談）"},
            {"time": "19:00", "name": "Le Wine Bar",
             "folder": "餐酒_results/8_Le Wine Bar"},
        ],
    },
]


# ── Image helpers ────────────────────────────────────────────────────────────

def crop_to_9_16(pil_img: Image.Image) -> Image.Image:
    """Center-crop and resize to W×H (1080×1920)."""
    iw, ih = pil_img.size
    if iw / ih > W / H:           # too wide → crop sides
        new_w = int(ih * W / H)
        left  = (iw - new_w) // 2
        pil_img = pil_img.crop((left, 0, left + new_w, ih))
    else:                          # too tall → crop top/bottom
        new_h = int(iw * H / W)
        top   = (ih - new_h) // 2
        pil_img = pil_img.crop((0, top, iw, top + new_h))
    return pil_img.resize((W, H), Image.LANCZOS)


def add_text_overlay(pil_img: Image.Image, time_str: str, name_str: str) -> np.ndarray:
    """Composite a gradient bar + time + name onto the image."""
    # Gradient overlay (transparent → dark at bottom)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    bar_start = H - 320
    for y in range(bar_start, H):
        alpha = int(200 * (y - bar_start) / (H - bar_start))
        draw_ov.line([(0, y), (W, y)], fill=(0, 0, 20, alpha))

    base = pil_img.convert("RGBA")
    base = Image.alpha_composite(base, overlay).convert("RGB")
    draw = ImageDraw.Draw(base)

    # Load fonts (fallback to default if font missing)
    try:
        f_time = ImageFont.truetype(FONT_PATH, 88, index=0)
        f_name = ImageFont.truetype(FONT_PATH, 54, index=0)
    except Exception:
        f_time = ImageFont.load_default()
        f_name = ImageFont.load_default()

    # Time label (golden)
    draw.text((70, bar_start + 30),  time_str, font=f_time, fill=(255, 210, 50))
    # Place name (white)
    draw.text((70, bar_start + 140), name_str, font=f_name, fill=(255, 255, 255))

    return np.array(base)


# ── Clip builders ────────────────────────────────────────────────────────────

def make_stop_clips(stop: dict) -> list:
    """Return a list of ImageClips for one itinerary stop."""
    folder  = os.path.join(DATA_DIR, stop["folder"])
    photos  = sorted(glob.glob(os.path.join(folder, "photo_*.jpg")))[:PHOTOS_PER_STOP]

    if not photos:
        print(f"  ⚠ No photos found in: {folder}")
        blank = np.zeros((H, W, 3), dtype=np.uint8)
        return [ImageClip(blank, duration=PHOTO_DUR)]

    clips = []
    for path in photos:
        pil_img = crop_to_9_16(Image.open(path).convert("RGB"))
        frame   = add_text_overlay(pil_img, stop["time"], stop["name"])
        clip    = (
            ImageClip(frame, duration=PHOTO_DUR)
            .with_fps(FPS)
            .with_effects([vfx.FadeIn(FADE_DUR), vfx.FadeOut(FADE_DUR)])
        )
        clips.append(clip)
    return clips


def make_route_video(route: dict) -> None:
    print(f"\n── {route['title']} ──")
    all_clips = []
    for stop in route["stops"]:
        print(f"  {stop['time']} {stop['name']}")
        all_clips.extend(make_stop_clips(stop))

    final = concatenate_videoclips(all_clips, method="compose")

    # BGM (optional)
    if os.path.exists(BGM_PATH):
        audio = (
            AudioFileClip(BGM_PATH)
            .subclipped(0, final.duration)
            .with_effects([afx.AudioFadeIn(1.0), afx.AudioFadeOut(2.0)])
        )
        final = final.with_audio(audio)
    else:
        print("  (no BGM — place bgm.mp3 next to this script to add music)")

    out_path = os.path.join(OUTPUT_DIR, route["output"])
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
    for route in ROUTES:
        make_route_video(route)
    print("\nDone! All videos saved to:", OUTPUT_DIR)
