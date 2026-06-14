"""
心斎橋 禅園 Instagram Reels動画生成スクリプト
サイズ: 1080x1920 (縦型)
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from moviepy import ImageClip, concatenate_videoclips, CompositeVideoClip, ColorClip

IMG_DIR = os.path.join(os.path.dirname(__file__), "images")
OUT_PATH = os.path.join(os.path.dirname(__file__), "output", "zenen_reel.mp4")
W, H = 1080, 1920
FPS = 30

FONT_PATH = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"

# シーン定義: (ファイル名, メインテロップ, サブテロップ, 表示秒数, ズーム方向)
SCENES = [
    ("_MG_1135.jpg",       "心斎橋 禅園",              "大人の夜が、ここから始まる。",    4.5, "in"),
    ("_MG_0912a修.jpg",    "特別な空間",                "静かに、ゆったりと。",             4.0, "in"),
    ("_MG_0983a修.jpg",    "上質な時間",                "光と影が織りなす、大人の夜。",     4.0, "out"),
    ("_MG_0989修.jpg",     "厳選されたお酒",            "職人が選び抜いた銘酒の数々。",     4.0, "in"),
    ("DSC07395.jpg",       "コース料理",                "素材の声を聴く、繊細な一皿。",     4.5, "in"),
    ("DSC00416.jpg",       "大人デートならここ",        "心斎橋で、忘れられない夜を。",     4.5, "out"),
    ("_MG_1126.jpg",       "日本の美意識",              "細部に宿る、本物のこだわり。",     4.0, "in"),
    ("心斎橋QR文字入り.jpg", "",                         "",                                 4.0, "none"),
]

FADE_DURATION = 0.6


def crop_to_portrait(img: Image.Image) -> Image.Image:
    """画像を1080x1920にトリム（中央クロップ）"""
    src_w, src_h = img.size
    target_ratio = W / H
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        # 横が広い → 高さ基準でリサイズして横をクロップ
        new_h = H
        new_w = int(src_w * (H / src_h))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - W) // 2
        img = img.crop((left, 0, left + W, H))
    else:
        # 縦が長い → 幅基準でリサイズして縦をクロップ
        new_w = W
        new_h = int(src_h * (W / src_w))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        top = (new_h - H) // 2
        img = img.crop((0, top, W, top + H))
    return img


def add_vignette(img: Image.Image, strength: float = 0.5) -> Image.Image:
    """周辺減光（ビネット）効果を追加"""
    arr = np.array(img).astype(np.float32)
    ys, xs = np.mgrid[0:H, 0:W]
    cx, cy = W / 2, H / 2
    dist = np.sqrt(((xs - cx) / (W / 2)) ** 2 + ((ys - cy) / (H / 2)) ** 2)
    vignette = 1 - np.clip(dist * strength, 0, 1)
    vignette = vignette[:, :, np.newaxis]
    arr = arr * vignette
    return Image.fromarray(arr.astype(np.uint8))


def draw_text_on_frame(img: Image.Image, main_text: str, sub_text: str) -> Image.Image:
    """テロップを描画"""
    img = img.copy()
    draw = ImageDraw.Draw(img)

    if not main_text and not sub_text:
        return img

    # 半透明の黒帯（下部）
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    band_top = H - 500
    for y in range(band_top, H):
        alpha = int(180 * (y - band_top) / (H - band_top))
        ov_draw.line([(0, y), (W, y)], fill=(0, 0, 0, alpha))

    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # メインテロップ（大きめ、ゴールド系）
    if main_text:
        font_size = 88
        try:
            font = ImageFont.truetype(FONT_PATH, font_size)
        except Exception:
            font = ImageFont.load_default()

        # 影
        bbox = draw.textbbox((0, 0), main_text, font=font)
        tw = bbox[2] - bbox[0]
        x = (W - tw) // 2
        y = H - 380
        draw.text((x + 3, y + 3), main_text, font=font, fill=(0, 0, 0, 180))
        draw.text((x, y), main_text, font=font, fill=(220, 190, 120))  # ゴールド

    # サブテロップ
    if sub_text:
        font_size_sub = 46
        try:
            font_sub = ImageFont.truetype(FONT_PATH, font_size_sub)
        except Exception:
            font_sub = ImageFont.load_default()

        bbox2 = draw.textbbox((0, 0), sub_text, font=font_sub)
        tw2 = bbox2[2] - bbox2[0]
        x2 = (W - tw2) // 2
        y2 = H - 260
        draw.text((x2 + 2, y2 + 2), sub_text, font=font_sub, fill=(0, 0, 0, 160))
        draw.text((x2, y2), sub_text, font=font_sub, fill=(240, 230, 210))  # クリーム白

    # 細いゴールドライン（装飾）
    if main_text:
        line_y = H - 415
        line_w = 200
        lx = (W - line_w) // 2
        draw.rectangle([lx, line_y, lx + line_w, line_y + 1], fill=(180, 150, 80))

    return img


def make_zoom_frames(base_img: Image.Image, duration: float, direction: str, fps: int) -> np.ndarray:
    """ゆっくりしたケンバーンズ（ズーム）効果フレームを生成"""
    n = int(duration * fps)
    arr = np.array(base_img)
    frames = []

    zoom_range = 0.06  # ±6%のズーム

    for i in range(n):
        t = i / max(n - 1, 1)

        if direction == "in":
            scale = 1.0 + zoom_range * (1 - t)  # 大→小（ズームアウト方向）
        elif direction == "out":
            scale = 1.0 + zoom_range * t  # 小→大（ズームイン方向）
        else:
            scale = 1.0

        crop_w = int(W / scale)
        crop_h = int(H / scale)
        ox = (W - crop_w) // 2
        oy = (H - crop_h) // 2

        frame = Image.fromarray(arr)
        frame = frame.crop((ox, oy, ox + crop_w, oy + crop_h))
        frame = frame.resize((W, H), Image.LANCZOS)
        frames.append(np.array(frame))

    return np.array(frames)


def build_scene(img_path: str, main_text: str, sub_text: str, duration: float, direction: str):
    """1シーン分のClipを生成"""
    img = Image.open(img_path).convert("RGB")
    img = crop_to_portrait(img)
    # 明度をやや落として高級感を演出
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(0.88)
    # ビネット
    img = add_vignette(img, strength=0.45)
    # テロップ
    img = draw_text_on_frame(img, main_text, sub_text)

    frames = make_zoom_frames(img, duration, direction, FPS)

    def make_frame(t):
        idx = min(int(t * FPS), len(frames) - 1)
        return frames[idx]

    clip = ImageClip(np.array(img), duration=duration)
    clip = clip.with_effects([])

    # フレームベースで作成
    from moviepy import VideoClip
    video = VideoClip(make_frame, duration=duration)
    video = video.with_fps(FPS)

    # フェードイン/アウト
    from moviepy.video.fx import FadeIn, FadeOut
    video = video.with_effects([FadeIn(FADE_DURATION), FadeOut(FADE_DURATION)])

    return video


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    clips = []
    for fname, main_t, sub_t, dur, zoom in SCENES:
        fpath = os.path.join(IMG_DIR, fname)
        print(f"Processing: {fname}")
        clip = build_scene(fpath, main_t, sub_t, dur, zoom)
        clips.append(clip)

    final = concatenate_videoclips(clips, method="compose")
    print(f"Rendering to {OUT_PATH} ...")
    final.write_videofile(
        OUT_PATH,
        fps=FPS,
        codec="libx264",
        audio=False,
        preset="slow",
        ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"],
    )
    print("Done!")


if __name__ == "__main__":
    main()
