import numpy as np
from scipy import interpolate
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import util
import XafsData
from typing import Callable
from importlib import import_module

__version__ = "0.0.4"


_logger = util.getLogger(__name__, level=util.DEBUG)


class Saxs1dProfile:
    def __init__(self, r: np.ndarray, i: np.ndarray):
        self.__r: np.ndarray = r
        self.__i: np.ndarray = i
        return

    @classmethod
    def load_csv(
        cls, path: str, *, delimiter: str = ",", skiprows: int = 4
    ) -> "Saxs1dProfile":
        try:
            table = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)
        except:
            table = np.loadtxt(
                path, delimiter=delimiter, skiprows=skiprows, encoding="cp932"
            )
        return cls(table[:, 0], table[:, 1])

    @property
    def r(self):
        return self.__r

    @property
    def i(self):
        return self.__i


class SaxsSeries:
    """
    attributes
    ----------
    dir : str
        path to directory containing csv files
    with_stdinfo : bool
        true if q2r and r2q are set by loadStdinfo
    q2r : Callable
        q2r(q[nm^-1]) -> r[px]
    r2q : Callable
        r2q(r[px]) -> q[nm^-1]
    r : np.ndarray
        1d array of radial coordinate[px] of the data (common to all files)
    i : np.ndarray
        2d array of intensity data (i[n_file, r.size])
    """

    def __init__(self, dir: str):
        self.dir = os.path.join(os.getcwd(), dir)
        self.with_stdinfo = False
        filePaths = [
            os.path.join(self.dir, name)
            for name in util.listFiles(self.dir, ext=".csv")
        ]
        files = [Saxs1dProfile.load_csv(f) for f in filePaths]
        self.r = files[0].r
        self.i = np.array([f.i for f in files], dtype=float)
        _logger.info(f"loaded {len(filePaths)} files from {self.dir}")
        return

    def loadStdinfo(self, relative_path):
        path = os.path.join(self.dir, relative_path)
        mod_std = import_module(path)
        self.q2r: Callable = mod_std.q2r
        self.r2q: Callable = mod_std.r2q
        self.with_stdinfo = True
        return

    def heatmap(
        self,
        ax: Axes,
        *,
        uselog: bool = True,
        y: np.ndarray = np.array([]),
        y_label: str = "file number",
        levels: int = 100,
        cmap: str = "rainbow",
        show_colorbar: bool = False,
        set_q_axis: bool = False,
    ) -> Axes:
        """plot heatmap of i[file, r] on given ax"""
        i = np.log(self.i) if uselog else self.i
        if y.size == 0:
            y = np.arange(i.shape[0])
        elif y.size != i.shape[0]:
            raise ValueError("invalid y size")
        contf = ax.contourf(self.r, y, i, levels=levels, cmap=cmap)
        if show_colorbar:
            cb_label = "ln(I)" if uselog else "I"
            plt.colorbar(contf).set_label(cb_label)
        ax.set_xlabel("r[px]")
        ax.set_ylabel(y_label)
        if set_q_axis and self.with_stdinfo:
            ax_d = ax.secondary_xaxis("top", functions=(self.r2q, self.q2r))
            ax_d.set_xlabel("q[nm^-1]")
        return ax

    def savefig(
        self,
        fig: Figure,
        *,
        dist: str = "",
        overwrite: bool | None = None,
        dpi: int = 300,
    ) -> str | None:
        """save figure"""
        if dist == "":
            dist = self.dir + ".png"
        if os.path.exists(dist):
            if overwrite is None:
                overwrite = input(f"{dist} already exists. overwrite? (y/n)") == "y"
            if not overwrite:
                print("file name conflict. aborted.")
                return None
        fig.savefig(dist, dpi=dpi)
        return dist


class DafsData(SaxsSeries):
    """
    attributes
    ----------
    xafsfile : XafsData
        xafs file includes energy and i0 at each file
    name : str
        sample name fetched from xafsfile
    i0 : np.ndarray
        i0 at each file
    energy : np.ndarray
        energy at each file
    r : np.ndarray (n,2)

    """

    def __init__(self, dir: str, xafsfile: str):
        """load xafs file and fetch i0 and energy"""
        super().__init__(dir)
        xafsfile = os.path.join(dir, xafsfile)
        self.xafsfile = XafsData.XafsData(xafsfile, cols=[3, 4])
        self.name = self.xafsfile.sampleinfo.split(" ")[0]
        if self.xafsfile.energy.size != self.i.shape[0]:
            raise ValueError("inconsistent number of files. incorrect xafs file ?")
        self.i0: np.ndarray = self.xafsfile.data
        self.energy: np.ndarray = self.xafsfile.energy
        self.n_e: int = self.energy.size

    def loadStdinfo(self, relative_path):
        """load stdinfo and set q2r and r2q interpolator
        arguments
        ---------
        relative_path : str
            path to stdinfo csv file from self.dir
            the file should have 5 columns: e, m, b, r_bottom, r_top and 1 row header
        """
        path = os.path.join(self.dir, relative_path)
        stdinfo = np.loadtxt(path, delimiter=",", skiprows=1)
        e: np.ndarray = stdinfo[:, 0]  # [keV]
        m: np.ndarray = stdinfo[:, 1]  # [nm^-1/px]
        b: np.ndarray = stdinfo[:, 2]  # [nm^-1]
        # r->qでデータを作ることでデータ範囲外のqに対してnanを返すようにする
        # XXX 原理的には係数mはeに反比例(?)するのでそれを使ったほうが良い(?)
        r_botttom: np.ndarray = stdinfo[:, 3]
        r_top: np.ndarray = stdinfo[:, 4]
        n_r = 3
        r = np.array(
            [np.linspace(r_botttom[i], r_top[i], n_r) for i in range(e.size)]
        ).flatten()
        q = np.empty_like(r)
        qe = np.empty((r.size, 2))
        re = np.empty((r.size, 2))
        for i in range(n_r):
            for j in range(e.size):
                ij = i * e.size + j
                q[ij] = m[j] * r[ij] + b[j]  # [nm^-1]
                qe[ij] = [q[ij], e[j]]
                re[ij] = [r[ij], e[j]]

        self.q2r = interpolate.LinearNDInterpolator(qe, r, fill_value=np.nan)
        self.r2q = interpolate.LinearNDInterpolator(re, q, fill_value=np.nan)

        self.with_stdinfo = True

        return

    def heatmap(
        self, ax: Axes, *, uselog: bool = False, y: np.ndarray = np.array([]), **kwargs
    ) -> Axes:
        ax = super().heatmap(
            ax, uselog=uselog, y=self.energy, y_label="energy[keV]", **kwargs
        )
        return ax

    def q_slice(self, q: float = np.nan) -> np.ndarray:
        """fetch e-i from all files at given q[nm^-1]
        returns
        -------
        i : np.ndarray
            i[e,q] for all e in self.energy
        """
        ret = np.empty_like(self.energy)
        arr_q = np.array(q).repeat(self.energy.size)
        r = self.q2r(arr_q, self.energy)
        for i, e in enumerate(self.energy):
            if np.isnan(r[i]):
                ret[i] = np.nan
            else:
                ret[i] = np.interp(self.q2r(q, e), self.r, self.i[i])
        return ret