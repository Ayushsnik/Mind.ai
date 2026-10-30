"""Generate a synthetic 'phone photo of a ruled journal page' for OCR tests."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

TEXT = [
    "Today the DBMS exam finally happened.",
    "I was nervous in the morning but the paper",
    "was easier than I expected.",
    "",
    "In the evening I went to the gym with Rohan",
    "and felt much calmer after the workout.",
]


def make_page(path, angle=4.0, shadow=True, seed=0):
    rng = np.random.default_rng(seed)
    W, H, line_h, top = 1400, 1100, 70, 140
    img = Image.new("L", (W, H), 238)
    d = ImageDraw.Draw(img)
    for y in range(top + 52, H - 40, line_h):          # notebook rules
        d.line([(40, y), (W - 40, y)], fill=175, width=2)
    d.line([(120, 0), (120, H)], fill=170, width=2)    # margin line
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
    for i, t in enumerate(TEXT):
        d.text((150, top + i * line_h), t, font=font, fill=35)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=238)
    arr = np.asarray(img).astype(np.float32)
    if shadow:   # dark corner like a hand/phone shadow
        yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
        arr *= 0.55 + 0.45 * (xx / arr.shape[1]) ** 0.6
    arr += rng.normal(0, 6, arr.shape)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6)).save(path)
    return "\n".join(TEXT)


if __name__ == "__main__":
    make_page("tests/assets/ruled_page.png")
