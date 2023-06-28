"""module for storing SAXS experiment data"""

import numpy as np
from util import Callable
import util
import SaxsProfile as sp

_logger = util.getLogger(__name__, util.DEBUG)


class SaxsExperiment:
    def __init__(self):
        self.__std: SaxsProfile = None  # type: ignore
        self.__data: list[sp.SaxsProfile] = []
        self.__mask: np.ndarray = None  # type: ignore
        self.__converter: Callable[[np.ndarray], np.ndarray] = None  # type: ignore
        self.__center: tuple[float, float] = None  # type: ignore
        return

    def addStd(self, src: str, *, nthpeak: int = 2, peak_q: float):
        self.__std = sp.loadTiff(src)
        self.__std.integrate()
        self.__mask = self.__std.mask
        self.__center = self.__std.center
        self.__converter = self.__std.generateConverter(nthpeak, peak_q)

    def addData(self, src: str):
        data = sp.loadTiff(src)
        data.mask = self.__mask
        data.center = self.__center
        data.integrate()
        data.convertToIq(converter=self.__converter)
        self.__data.append(data)
