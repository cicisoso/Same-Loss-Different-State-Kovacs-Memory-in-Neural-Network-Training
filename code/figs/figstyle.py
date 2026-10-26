# Shared style for the Kovacs-memory figures (matplotlib; Nimbus Sans as the Helvetica-metric font on the server).
# Advanced Science: axis labels and symbols 10-12 pt at reproduction size, one font size per figure, panel letters 12 pt
# bold sans-serif; figures are drawn at their final width (17.8 cm double column), so all text is 10 pt.
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mt
try:                       # panel-alignment gate of the nature-figure skill (QA helper, not redistributed with the code)
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "axes.spines.right": False,
    "axes.spines.top": False, "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6, "ytick.minor.width": 0.6, "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0, "ytick.minor.size": 2.0, "lines.linewidth": 1.4, "legend.frameon": False,
    "legend.handlelength": 1.2, "legend.handletextpad": 0.4, "legend.labelspacing": 0.25, "legend.borderaxespad": 0.3,
    "axes.labelpad": 3.0, "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
    "mathtext.fontset": "stixsans",   # Greek nu in Nimbus Sans is indistinguishable from a Latin v
})
C = {"hot": "#B64342", "hot_soft": "#E9A6A1", "cold": "#0F4D92", "cold_soft": "#9DB9DD", "ref": "#4D4D4D",
     "grey": "#A8A8A8", "light": "#D8D8D8", "kov": "#6A3D9A", "teal": "#42949E", "gold": "#B8860B", "green": "#2E7D32"}
W2 = 7.0                                                               # double-column width (17.8 cm) in inches


def label(ax, s, dx_pt=-40, dy_pt=8, **_):
    """panel letter at a fixed physical offset from the axes' top-left corner (aligned across panels of any width)"""
    tr = ax.transAxes + mt.ScaledTranslation(dx_pt / 72, dy_pt / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, s, transform=tr, fontsize=12, fontweight="bold", va="bottom", ha="left")


def save(fig, stem, **manifest):
    """alignment gate on the final layout (manifest: exclude_axes for insets, exemptions with reasons), then export"""
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(fig, json_out=f"{stem}.alignment.json", overlay_svg=f"{stem}.alignment.svg",
                                           tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, **manifest)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight"); fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
