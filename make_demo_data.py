"""Generate deterministic, clearly synthetic signatures for ten demo identities."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "data" / "raw"
NAMES = [
    "Alex Morgan",
    "Bailey Carter",
    "Casey Jordan",
    "Drew Parker",
    "Elliot Taylor",
    "Finley Harper",
    "Gray Cameron",
    "Hayden Quinn",
    "Indigo Riley",
    "Jamie Rowan",
]


def _font_paths() -> list[Path]:
    """Find available TrueType fonts on common Windows and Linux installs."""
    candidates = [
        Path("C:/Windows/Fonts/segoesc.ttf"),
        Path("C:/Windows/Fonts/seguisb.ttf"),
        Path("C:/Windows/Fonts/BRUSHSCI.TTF"),
        Path("C:/Windows/Fonts/ITCEDSCR.TTF"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
        Path("/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf"),
    ]
    return [path for path in candidates if path.is_file()]


def generate_demo_data(output_dir: Path, samples_per_person: int = 24) -> int:
    """Render font/stroke variants for ten fake identities, returning file count."""
    if samples_per_person < 3:
        raise ValueError("Use at least 3 samples per person for stratified splits.")
    random.seed(42)
    rng = np.random.default_rng(42)
    fonts = _font_paths()
    total = 0
    for person_index, name in enumerate(NAMES):
        person_dir = output_dir / f"person_{person_index + 1:02d}" / "genuine"
        person_dir.mkdir(parents=True, exist_ok=True)
        for sample_index in range(samples_per_person):
            canvas = Image.new("L", (520, 180), color=255)
            draw = ImageDraw.Draw(canvas)
            font_size = random.randint(42, 72)
            if fonts:
                font = ImageFont.truetype(str(fonts[person_index % len(fonts)]), font_size)
            else:
                font = ImageFont.load_default()
            x = random.randint(24, 48)
            y = random.randint(28, 72)
            ink = random.randint(10, 70)
            draw.text((x, y), name, fill=ink, font=font, stroke_width=random.choice([0, 1]))

            # Individualized flourish and underline make demo classes distinguishable.
            base_x = x + random.randint(30, 110)
            base_y = y + random.randint(45, 80)
            flourish_width = 60 + person_index * 8
            points = []
            for step in range(45):
                t = step / 44
                curve_y = base_y + math.sin(t * math.pi * (1 + person_index % 3)) * (
                    5 + person_index
                )
                points.append(
                    (base_x + t * flourish_width, curve_y + rng.normal(0, 1.2))
                )
            draw.line(points, fill=ink, width=random.choice([1, 2, 3]))
            if person_index % 2 == 0:
                line_y = min(160, base_y + random.randint(18, 35))
                draw.line(
                    [(x, line_y), (x + random.randint(230, 390), line_y - random.randint(0, 12))],
                    fill=ink,
                    width=1,
                )

            angle = random.uniform(-4, 4)
            canvas = canvas.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)
            pixels = np.asarray(canvas, dtype=np.int16)
            noise = rng.normal(0, 2.0, pixels.shape)
            canvas = Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))
            file_path = person_dir / f"signature_{sample_index + 1:03d}.png"
            canvas.save(file_path)
            total += 1
    return total


def main() -> None:
    """Parse optional generation settings and create the demo dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-person", type=int, default=24)
    args = parser.parse_args()
    count = generate_demo_data(args.output, args.samples_per_person)
    print(
        f"Generated {count} synthetic images for {len(NAMES)} identities in "
        f"{args.output.resolve()}"
    )
    print("Synthetic demo signatures are not real handwriting or a benchmark.")


if __name__ == "__main__":
    main()
