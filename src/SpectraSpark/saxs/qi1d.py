import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
import os
import warnings

class Saxs1d:
    """1次元のSaxsデータを扱うクラス"""

    __DEFAULT_DELIMITER = ","

    def __init__(self, i: np.ndarray, q: np.ndarray):
        self.__intensity = i
        self.__q = q
        return

    @property
    def i(self) -> np.ndarray:
        """scattering intensity"""
        return self.__intensity

    @property
    def q(self) -> np.ndarray:
        """magnitude of scattering vector [nm^-1]"""
        return self.__q

    @classmethod
    def load(cls, src: str) -> "Saxs1d":
        """csvファイルから読み込む
        P2M.toChiFile()で出力したファイル(ヘッダ3行)を読み込む
        第1列はq[nm^-1]
        """
        data = np.loadtxt(src, delimiter=cls.__DEFAULT_DELIMITER, skiprows=3)
        return Saxs1d(data[:, 1], data[:, 0])

    @classmethod
    def loadMatFile(cls, src: str, usecol: int) -> "Saxs1d":
        """データ列が複数のファイルから読み込む"""
        data = np.loadtxt(
            src, usecols=(0, usecol), delimiter=cls.__DEFAULT_DELIMITER, skiprows=1
        )
        return Saxs1d(data[:, 1], data[:, 0])

    def guinierRadius(self):
        """ギニエ半径を求める"""
        raise NotImplementedError

    def integratedIntensity(self):
        """積分強度を求める"""
        raise NotImplementedError


class Saxs1dSeries(Saxs1d):
    """時分割のSAXSデータ系列を扱うクラス"""

    __DEFAULT_DELIMITER = ","

    def __init__(self, src: str='', *, i: np.ndarray, q: np.ndarray):
        if src:
            self.loadMatFile(src)
        elif q.shape[0] == i.shape[1]:
            super().__init__(i, q)
        else:
            raise ValueError("src path or corresponding i and q needed")

    @classmethod
    def loadMatFile(cls, src: str) -> "Saxs1dSeries":
        data = np.loadtxt(src, delimiter=cls.__DEFAULT_DELIMITER)
        q = data[:, 0]
        i = data[:, 1:].T  # i[k]がk番目のプロファイル
        print(f"{i.shape[0]} profiles along {q.shape[0]} q values loaded")
        return cls(i=i, q=q)

    @classmethod
    def load(cls, src: str) -> "Saxs1dSeries":
        """csvファイルを読み込んでSaxs1dSeriesを返す

        qi2d.series_integrateで出力したファイルを読み込む
        """
        return cls.loadMatFile(src)

    def load_temperature(self, src: str, *, skiprows=1, usecol=4, delimiter=","):
        """温度データを読み込む"""
        values = np.loadtxt(src, delimiter=delimiter, skiprows=skiprows, usecols=usecol)
        if len(values) < self.i.shape[0]:
            raise ValueError("Temperature data size not match")
        self.__temperature = values[:self.i.shape[0]]
        return

    @property
    def t(self) -> np.ndarray:
        """temperature history"""
        return self.__temperature

    def peakHistory(self, q_min: float, q_max: float) -> np.ndarray:
        """ピーク強度の時系列を求める"""
        raise NotImplementedError

    def heatmap(
            self,
            fig: Figure,
            ax: Axes,
            *,
            logscale: bool = True,
            x_label="$q$ [nm$^{-1}$]",
            x_lim=(np.nan, np.nan),
            y_label: str = "file number",
            y_ticks = [],
            y_tick_labels: list[str] = [],
            y_lim=(0, None),
            v_min=np.nan,
            v_max=np.nan,
            cmap: str | Colormap = "jet",
            show_colorbar: bool = True,
            extend: str = "min",
            cbar_fraction: float = 0.02,
            cbar_pad: float = 0.08,
            cbar_aspect: float = 50,
            **kwargs
    ):
        """ヒートマップを保存する"""
        q = self.q.copy()
        val = self.i.copy()
        if logscale:
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            val = np.log10(val)
            warnings.resetwarnings()
            isvalid = (np.isfinite(val).sum(axis=0)==val.shape[0])
            q = q[isvalid]
            val = val[:, isvalid]

        idx = ((x_lim[0] < q) * (q < x_lim[1]))
        val = val[y_lim[0]:y_lim[1], idx]
        q = q[idx]
        y = np.arange(1, val.shape[0] + 1)
        v_min = np.nanmin(val) if np.isnan(v_min) else v_min  # TODO: bag fix
        v_max = np.nanmax(val) if np.isnan(v_max) else v_max
        levels = np.linspace(v_min, v_max, 256)
        cs = ax.contourf(q, y, val, levels=levels, cmap=cmap, extend=extend)

        if show_colorbar:
            cbar = fig.colorbar(
                cs, ax=ax,
                fraction=cbar_fraction, pad=cbar_pad, aspect=cbar_aspect,
                **kwargs
            )
            cbar.set_label(r"$\log[I(q)]\;[a.u.]$" if logscale \
                           else r"$I[q]\;[a.u.]$")

        ax.set_xlabel(x_label)
        if len(y_ticks) != 0:
            ax.set_yticks(y_ticks, y_tick_labels)
        if y_label != "":
            ax.set_ylabel(y_label)
        return

    @classmethod
    def saveHeatmap(cls, src, *,
                    figsize=(6, 6), q_min=1, q_max=10,
                    v_min=np.nan, v_max=np.nan, overwrite=False,
                    **kwargs) -> str:
        dst = src.replace(".csv", ".png")
        if not overwrite and os.path.exists(dst):
            raise FileExistsError(f"{dst} already exists")
        title = os.path.basename(dst).split(".")[0]

        obj = cls.loadMatFile(src)
        fig, ax = plt.subplots(figsize=figsize)
        obj.heatmap(
            fig, ax, x_lim=(q_min, q_max), v_min=v_min, v_max=v_max, **kwargs
        )

        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(dst, dpi=300)
        return dst
