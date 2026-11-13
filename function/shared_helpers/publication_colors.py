"""Colour palettes for accessible scientific figures."""

from __future__ import annotations

import colorsys


# Okabe-Ito colours are widely used for colour-blind-safe scientific figures.
# The extension colours retain strong luminance contrast for longer series.
NATURE_ACCESSIBLE_PALETTE = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
    "#332288",  # indigo
    "#AA4499",  # purple
    "#44AA99",  # teal
    "#117733",  # dark green
)


def distinct_publication_colors(count: int) -> tuple[str, ...]:
    """Return one deterministic, colour-blind-friendly colour per series.

    The first twelve colours are fixed accessible publication colours. For
    longer series, additional high-contrast colours are generated without
    repeating an earlier colour; line labels and markers remain available as
    additional visual identifiers in the plot legend.
    """
    if count < 0:
        raise ValueError("The number of requested colours cannot be negative.")
    if count == 0:
        return ()
    colors = list(NATURE_ACCESSIBLE_PALETTE[:count])
    for index in range(len(colors), count):
        hue = (index * 0.618033988749895) % 1.0
        lightness = 0.38 if index % 2 else 0.68
        red, green, blue = colorsys.hls_to_rgb(hue, lightness, 0.68)
        colors.append(f"#{round(red * 255):02X}{round(green * 255):02X}{round(blue * 255):02X}")
    return tuple(colors)
