import subprocess
import matplotlib as mpl
from os import PathLike

def mpl_usetex(_: bool = True):
    mpl.rcParams['text.usetex'] = bool(_)

def mpl_style(style: str = ""):
    style = style.casefold()
    style_lookup = {
        "": "classic",
        "default": "classic",
        "seaborn": "seaborn-v0_8-darkgrid",
    }
    mpl.style.use(style_lookup[style] if style in style_lookup else style)

def mpl_fontsize(size: int = 20):
    mpl.rcParams['font.size'] = int(size)

def mpl_render(path: str | PathLike | None = None):
    if path is None:
        mpl.pyplot.show()
    else:
        mpl.pyplot.savefig(path)

__all__ = ["mpl_usetex", "mpl_style", "mpl_fontsize", "mpl_render"]