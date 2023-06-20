import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # type: ignore
from matplotlib.figure import Figure
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .. import util

_logger = util.getLogger(__name__, util.DEBUG)

TMPDIR = "tmp/"

WINDOW_SIZE = (500, 750)
CANVAS_SIZE = (500, 600)

STATE_INIT = "init"
STATE_WAIT_AUTO_MASK = "wait_auto_mask"
STATE_WAIT_DETECT_CENTER = "wait_detect_center"
STATE_WAIT_SELECT_CENTER = "wait_select_center"
STATE_WAIT_INTEGRATE = "wait_integrate"


class FlushableFigCanvas:
    """canvas for matplotlib figure on tkinter"""

    def __init__(self, canvas: tk.Canvas):
        self.__canvas = canvas
        self.__canvas_packed = {}
        self.__fig_agg: FigureCanvasTkAgg = None  # type: ignore
        return

    def draw(self, figure: Figure) -> None:
        """refresh canvas and draw figure

        Parameters
        ----------
        figure : mlp.figure.Figure
            figure to draw
        """
        if self.__fig_agg is not None:
            self.__flush()
        self.__fig_agg = FigureCanvasTkAgg(figure, self.__canvas)
        self.__fig_agg.draw()
        widget = self.__fig_agg.get_tk_widget()
        if widget not in self.__canvas_packed:
            self.__canvas_packed[widget] = True
            widget.pack(side="top", fill="both", expand=1)
        return

    def __flush(self) -> None:
        """remove figure"""
        self.__fig_agg.get_tk_widget().forget()
        try:
            self.__canvas_packed.pop(self.__fig_agg.get_tk_widget())
        except Exception as e:
            _logger.error(f"error removing {self.__fig_agg}: {e}")
        plt.close("all")
        return

    def heatmap(self, data: np.ndarray) -> None:
        """flush canvas and draw heatmap"""
        fig = Figure()
        ax = fig.add_subplot(111)
        cmap: mpl.colormap.Colormap = mpl.colormaps.get_cmap("hot").copy()  # type: ignore
        cmap.set_bad("lime", alpha=1.0)
        im = ax.imshow(data, cmap=cmap)
        fig.colorbar(im, ax=ax)
        self.draw(fig)
        return

    def plot(self, x: np.ndarray, y: np.ndarray) -> None:
        """flush canvas and draw line plot"""
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        self.draw(fig)
        return
