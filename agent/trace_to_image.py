"""
Example:
chain of thought:
The user wants to find a storage option for 31 Nintendo Switch game cards.
I will use the search bar to find relevant products. The search bar has bid '258'.
I will fill it with "Nintendo Switch game card storage" and press Enter to initiate the search.

action:
fill('258', 'Nintendo Switch game card storage')
press('258', 'Enter')

number of actions taken: 2

intent:
Buy the least expensive red blanket (in any size) from "Blankets & Throws" category.

Simplified SOM:
[] [StaticText] [My Account]
[1] [A] [My Account]
[] [StaticText] [My Wish List]
[2] [A] [My Wish List]
[] [StaticText] [Sign In]
[3] [A] [Sign In]
[] [StaticText] [Welcome to One Stop Market]
[] [StaticText] [Create an Account]

"""

from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple
import textwrap


@dataclass
class AgentTraceStep:
    chain_of_thought: str
    action: str
    num_actions_taken: int
    intent: str
    simplified_som: str


def render_trace_to_image(
    trace: AgentTraceStep,
    width: int = 1280,
    height: int = 720,
    font_size: int = 18,
    padding: int = 40,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (30, 30, 30),
    label_color: Tuple[int, int, int] = (80, 80, 200),
) -> Image.Image:
    img = Image.new("RGB", (width, height), color=bg_color)  # type: ignore[arg-type]
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
        bold_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
        bold_font = font

    usable_width = width - 2 * padding
    chars_per_line = max(1, usable_width // (font_size * 6 // 10))
    y = padding

    def draw_section(label: str, body: str) -> None:
        nonlocal y
        if y + font_size > height - padding:
            return
        draw.text((padding, y), label, font=bold_font, fill=label_color)
        y += font_size + 6
        for raw_line in body.splitlines():
            for line in textwrap.wrap(raw_line, width=chars_per_line) or [""]:
                if y + font_size > height - padding:
                    return
                draw.text((padding, y), line, font=font, fill=text_color)
                y += font_size + 4
        y += font_size

    draw_section("Intent:", trace.intent)
    draw_section("Chain of Thought:", trace.chain_of_thought)
    draw_section("Action:", trace.action)
    draw_section(f"Actions Taken: {trace.num_actions_taken}", "")
    draw_section("Simplified SOM:", trace.simplified_som)

    return img


if __name__ == "__main__":
    trace = AgentTraceStep(
        chain_of_thought=(
            "The user wants to find a storage option for 31 Nintendo Switch game cards.\n"
            "I will use the search bar to find relevant products. The search bar has bid '258'.\n"
            'I will fill it with "Nintendo Switch game card storage" and press Enter to initiate the search.'
        ),
        action=(
            "fill('258', 'Nintendo Switch game card storage')\n"
            "press('258', 'Enter')"
        ),
        num_actions_taken=2,
        intent='Buy the least expensive red blanket (in any size) from "Blankets & Throws" category.',
        simplified_som=(
            "[] [StaticText] [My Account]\n"
            "[1] [A] [My Account]\n"
            "[] [StaticText] [My Wish List]\n"
            "[2] [A] [My Wish List]\n"
            "[] [StaticText] [Sign In]\n"
            "[3] [A] [Sign In]\n"
            "[] [StaticText] [Welcome to One Stop Market]\n"
            "[] [StaticText] [Create an Account]"
        ),
    )

    img = render_trace_to_image(trace, width=1280, height=720)
    out_path = "trace_render.png"
    img.save(out_path)
    print(f"Saved to {out_path}")
