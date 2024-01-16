import numpy as np
from scipy import signal, optimize
import re

__version__ = "0.1.1"


_EMPTY = np.array([])


def _split_by_spaces(line: str):
    return re.split(r" +", line.strip())


class XafsData:
    """XAFS data class
    attributes
    ----------
    fileformat : str
        file format of the data (now only 9809 is supported)
    src : str
        path to the data file
    beamline : str
    sampleinfo : str
    ring : str
    mono : str
    param : list[str]
    energy : np.ndarray
    data : np.ndarray | dict[str, np.ndarray]
        2d array of the data (data[n_point, n_columns])
        or dict of 1d array corresponding to each channel if dict `cols` is specified
    """

    def __init__(
        self,
        src: str,
        *,
        fileformat: str = "9809",
        cols: list[int] | dict[int, str] = [],
    ):
        if fileformat == "9809":
            self.fileformat = "9809"
            self.load_9809(src, cols=cols)
        else:
            raise ValueError(f"invalid fileformat: {fileformat}")
        return

    def load_9809(self, src: str, cols: list[int] | dict[int, str] = []):
        with open(src, "r") as f:
            lines = f.readlines()

        self.src = src
        self.beamline = lines[0].strip()
        self.sampleinfo = lines[1].strip()
        self.ring = lines[3].strip()
        self.mono = lines[4].strip()
        self.param = [lines[5].strip(), lines[6].strip()]
        if match := re.search(r"Points=( +)([0-9]+)", self.param[0]):
            n_points = int(match.group(2))
        else:
            raise ValueError(f"invalid file format: {src}\n  cannot find Points=...")
        if match := re.search(r"Block =( +)([0-9]+)", self.param[1]):
            n_blocks = int(match.group(2))
        else:
            raise ValueError(f"invalid file format: {src}\n  cannot find Block=...")

        self.__load_blocks([line.strip() for line in lines[9 : 9 + n_blocks]])
        if self.energy.size != n_points:
            raise ValueError(
                f"file may be damaged: {src}\n  energy.size:{self.energy.size} != n_points:{n_points}"
            )

        data_starts = 9 + n_blocks + 4
        channels = _split_by_spaces(lines[data_starts - 1])
        if cols == []:
            cols = list(range(len(channels)))
        if type(cols) == dict:
            self._data = np.loadtxt(
                src, skiprows=data_starts, usecols=list(cols.keys())
            )
            self.data = {
                key: self._data[:, i].view() for i, key in enumerate(cols.values())
            }
        elif hasattr(cols, "__iter__") and type(cols[0]) == int:
            self._data = np.loadtxt(src, skiprows=data_starts, usecols=cols)  # type: ignore
            self.data = self._data.view()
        else:
            raise ValueError(f"invalid cols: {cols}")

        if self._data.shape[0] != n_points:
            raise ValueError(
                f"file may be damaged: {src}\n  data.shape[1]:{self._data.shape[1]} != n_points:{n_points}"
            )

        return

    def __load_blocks(self, lines: list[str]):
        table = [_split_by_spaces(line) for line in lines]
        ls_energy = []
        for row in table:
            init_energy = float(row[1])
            fin_energy = float(row[2])
            num = int(row[5])
            ls_energy.append(np.linspace(init_energy, fin_energy, num))
        self.energy = np.concatenate(ls_energy)
        return


class DafsSpectrum:
    """DAFS spectrum class

    Attributes
    ----------
    energy : np.ndarray [eV]
    scattering : np.ndarray
        最大が1になるように正規化された散乱強度
    fluorescence : np.ndarray
        最大が1になるように正規化された蛍光強度
    mu : np.ndarray
        最大が1となるように正規化された吸収係数
    ttheta : np.ndarray
        scatteringに対応する散乱角(2θ)[deg]
    e0 : float
        吸収端のエネルギ[eV]
    """

    __N_EXCLUDE_FIT = 400
    __N_FIT = 20_000

    def __init__(
        self, energy, scattering, e0, fluorescence=_EMPTY, mu=_EMPTY, ttheta=_EMPTY
    ):
        self.__energy = energy
        self.__scattering = scattering
        self.__fluorescence = fluorescence
        self.__mu = mu
        self.__ttheta = ttheta
        self.__e0 = e0
        self.__fa_fit = _EMPTY
        self.__e_fit = _EMPTY
        self.__scattering_fit = _EMPTY
        return

    @property
    def energy(self) -> np.ndarray:
        return self.__energy

    @property
    def scattering(self) -> np.ndarray:
        return self.__scattering

    @property
    def fluorescence(self) -> np.ndarray:
        return self.__fluorescence

    @property
    def mu(self) -> np.ndarray:
        return self.__mu

    @property
    def ttheta(self) -> np.ndarray:
        return self.__ttheta

    @property
    def e0(self) -> float:
        return self.__e0

    @property
    def fa_fit(self) -> np.ndarray:
        return self.__fa_fit

    @property
    def e_fit(self) -> np.ndarray:
        return self.__e_fit

    @property
    def scattering_fit(self) -> np.ndarray:
        return self.__scattering_fit

    @staticmethod
    def __hilbert(array: np.ndarray) -> np.ndarray:
        return signal.hilbert(array).imag  # type: ignore

    @staticmethod
    def __mu2fa(mu: np.ndarray, e: np.ndarray) -> np.ndarray:
        """吸収係数から複素異常散乱項を求める
        異常散乱項は虚部の最大値を1に規格化した複素数の配列で返す
        """
        f_2dash = mu * e
        f_2dash = f_2dash / f_2dash.max()
        f_dash = DafsSpectrum.__hilbert(f_2dash)
        return f_dash + 1j * f_2dash

    @staticmethod
    def __fa2i(fa, *, a: float = 1, aoa: float = 1) -> np.ndarray:
        return a**2 * (1 + 2 * aoa * fa.real + aoa**2 * np.abs(fa) ** 2)

    @staticmethod
    def __fitError(params, i_exp, f_fit, n_exclude: int = 0) -> float:
        """フィッティングの誤差を返す"""
        n = len(i_exp)
        idx = np.abs(np.arange(n) - n // 2) > n_exclude // 2
        return np.sum(
            (i_exp[idx] - DafsSpectrum.__fa2i(f_fit, a=params[0], aoa=params[1])[idx])
            ** 2
        )

    def fit(self, n: int = -1, *, max_iter: int = 100, n_exclude: int = -1):
        """xafsスペクトルを抽出する
        nはフィッティングに用いる点の数, n_excludeは吸収端近傍でフィッティングから除く点の数
        """
        # TODO: 散乱強度の正規化
        #       - 蛍光の除去 <- おそらく定数倍を引けばいいが、その定数はどうやって求める?
        #       - 吸収の補正 <- 吸収係数の正規化(計数効率で真値に定数の"差"が乗る?)
        e_min, e_max = self.energy.min(), self.energy.max()
        n = n if n > -1 else self.__N_FIT
        n_exclude = n_exclude if n_exclude > -1 else self.__N_EXCLUDE_FIT
        e = np.linspace(e_min, e_max, n)
        i = np.interp(e, self.energy, self.scattering)

        # ここからフィッティング
        mu_ini, a_ini, aoa_ini = (e > self.e0).astype(float), 1, 1
        fa_ini = self.__mu2fa(mu_ini, e)
        fit = (
            fa_ini,
            optimize.minimize(
                fun=self.__fitError, x0=[a_ini, aoa_ini], args=(i, fa_ini)
            ).x,
        )
        fits = [fit]
        error = np.inf
        for _i in range(max_iter):
            a, aoa = fits[-1][1]
            f_2dash = fits[-1][0].imag
            f_dash = (np.sqrt((i / a**2) - (aoa * f_2dash) ** 2) - 1) / aoa
            f_2dash = -self.__hilbert(f_dash)
            f_2dash = f_2dash - f_2dash.min()  # 最小を0に
            fa_next = f_dash + 1j * f_2dash
            fits.append(
                (
                    fa_next,
                    optimize.minimize(
                        fun=self.__fitError, x0=[a, aoa], args=(i, fa_next, n_exclude)
                    ).x,
                )
            )
            _error = self.__fitError(fits[-1][1], i, fits[-1][0], n_exclude=0)
            if _error > error:
                break
            else:
                error = _error

        self.__fa_fit = fits[-1][0]
        self.__e_fit = e
        self.__scattering_fit = self.__fa2i(
            self.__fa_fit, a=fits[-1][1][0], aoa=fits[-1][1][1]
        )

        return self.e_fit, self.fa_fit, self.scattering_fit