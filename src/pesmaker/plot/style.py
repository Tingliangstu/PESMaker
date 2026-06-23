# Copyright 2026 Ting Liang and PESMaker development team
# This file is part of PESMaker.
#
# PESMaker is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PESMaker is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PESMaker. If not, see <https://www.gnu.org/licenses/>.
"""Shared plotting style helpers."""

from __future__ import annotations


def apply_plot_style() -> None:
    """Apply PESMaker's default publication-style plotting theme."""
    try:
        import seaborn as sns
    except ImportError:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-ticks")
        return
    sns.set_theme(
        style="ticks",
        context="notebook",
        font_scale=1.05,
        rc={
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.25,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        },
    )
