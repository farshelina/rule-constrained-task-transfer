"""Shared paper-wide visual palette.

Color semantics are global across plots:
- POSITIVE: improvement, strengthening, or evidence-supporting direction
- NEGATIVE: degradation, weakening, or adverse direction
- NEUTRAL: baseline, controls, and non-directional structure

Do not introduce additional categorical hues unless a new semantic role is
unavoidable. Use hatching, line style, marker shape, or labels first.
"""

from matplotlib.colors import LinearSegmentedColormap

POSITIVE = "#3F8C7A"
NEGATIVE = "#B05A4F"
NEUTRAL = "#7A838A"
INK = "#454B50"
WHITE = "#FFFFFF"

SIGNED_CMAP = LinearSegmentedColormap.from_list(
    "paper_signed_direction",
    [NEGATIVE, WHITE, POSITIVE],
    N=256,
)
