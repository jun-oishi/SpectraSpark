"""module for 2D SAXS profile"""

import numpy as np
import cv2
import os
import util
from typing import Callable

_logger = util.getLogger(__name__, level=util.DEBUG)
# _logger = util.getLogger(__name__, level=util.WARNING)


class _Masks(np.ndarray):
    """array-like class for masks
    support operations like numpy.ndarray
    value is always 0 or 1, 0 for masked pixel

    Methods
    -------
    append(new_mask: np.ndarray)
        append new mask and update value of self
    add_circle(center: tuple, radius: float, maskType: str = "out")
        add circle mask
    undo():
        delete last added mask
    """

    def __new__(cls, mold):
        value = np.ones_like(mold)
        self = np.asarray(value, dtype=bool).view(cls)
        self.__masks = []
        return self

    def __init__(self, mold):
        """initialize with mold for shape"""
        self.__masks = []
        return

    def __update(self) -> None:
        """update mask"""
        self[:] = self * 0 + 1
        for mask in self.__masks:
            self *= mask
        return

    def append(self, new_mask: np.ndarray) -> None:
        """append mask"""
        _logger.debug(f"append mask: shape={new_mask.shape} ,dtype={new_mask.dtype}")
        if new_mask.shape != self.shape:
            raise ValueError("invalid shape")
        self.__masks.append(new_mask)
        self *= new_mask
        return

    def __pop(self, index: int) -> None:
        """delete mask at the specified index"""
        self.__masks.pop(index)
        self.__update()
        return

    def add_circle(self, center: tuple, radius: float, maskType: str = "out") -> None:
        """add circle mask
        maskType: "out" or "in", default "out" for mask outside circle
        """
        dx = np.ones_like(self) * np.arange(self.shape[1]) - center[1]
        dy = np.ones_like(self) * np.arange(self.shape[0]).reshape(-1, 1) - center[0]
        dist = np.sqrt(dx**2 + dy**2)
        if maskType == "out":
            self.append(dist < radius)
        elif maskType == "in":
            self.append(dist > radius)
        else:
            raise ValueError("invalid mask")
        return

    def undo(self) -> None:
        """delete last added mask"""
        self.__pop(-1)
        return


GREEN = (0, 255, 0)  # BGR


class Saxs2dProfile:
    """SAXS 2D profile class

    Methods
    -------
    values(log:bool=True, showMaskAsNan:bool=True, showCenterAsNan:bool=False)
        return modified values
    """

    DEFAULT_MARK_COLOR = GREEN

    def __new__(cls):
        raise NotImplementedError(f"{cls} default initializer not implemented")

    def __init__(self, raw: np.ndarray):
        _logger.debug(f"initializing Saxs2dProfile with raw:{id(raw):x}, {raw.shape}")
        self.__raw: np.ndarray = raw
        self.__buf: np.ndarray = np.zeros_like(raw)
        self.__masks: _Masks = _Masks(self.__raw)
        self.__center: tuple = (np.nan, np.nan)
        _logger.debug(f"id(self.__raw): {id(self.__raw)}")

    @property
    def shape(self) -> tuple[int, int]:
        return self.__raw.shape

    @classmethod
    def __internal_new(cls):
        return object.__new__(cls)

    @classmethod
    def load_tiff(cls, path: str) -> "Saxs2dProfile":
        """load profile from tiff file

        Parameters
        ----------
        path: str
            path to tiff file

        Returns
        -------
        Saxs2dProfile
        """
        ret = cls.__internal_new()
        if not os.path.exists(path):
            raise FileNotFoundError("")
        if path[-4:] != ".tif":
            raise ValueError("invalid file type")
        ret.__init__(cv2.imread(path, cv2.IMREAD_UNCHANGED))
        _logger.debug(f"max: {ret.__raw.max()}, min: {ret.__raw.min()}")
        return ret

    def values(
        self,
        log: bool = True,
        showMaskAsNan: bool = True,
        showCenterAsNan: bool = False,
    ) -> np.ndarray:
        """get modified values

        Parameters
        ----------
        log: bool, default True
            if True, return ln(values), set nan for value <= 0
        showMaskAsNan: bool, default True
            if True, set nan for masked pixel
        showCenterAsNan: bool, default False
            if True, draw center mark(tilted cross) with nan value

        Returns
        -------
        np.ndarray
            the shape is same as raw data
        """
        self.__buf = self.__raw.copy()
        if showMaskAsNan:
            self.__buf = self.__buf.astype(np.float32)
            self.__buf[self.__masks == 0] = np.nan
        else:
            self.__buf *= self.__masks
        if log:
            self.__log()
        if showCenterAsNan:
            self.__draw_center()

        return self.__buf

    @property
    def center(self) -> tuple[float, float]:
        return self.__center

    @center.setter
    def center(self, center: tuple[float, float]):
        try:
            if len(center) != 2:
                raise TypeError("")
        except TypeError:
            raise TypeError("center must be array-like of 2 floats")

        shape = self.__raw.shape
        if (
            (0 < center[0])
            and (center[0] < shape[0])
            and (0 < center[1])
            and (center[1] < shape[1])
        ):
            self.__center = center
        else:
            raise ValueError("center must be in the range of raw data")

    def save(
        self,
        path: str,
        *,
        overwrite: bool = False,
        log: bool = True,
        color: bool = True,
        showMask: bool = True,
        showCenter: bool = False,
    ) -> int:
        """save modified image
        returns 0 if success

        Parameters
        ----------
        path: str
            path to save
        overwrite: bool, default False
            if True, overwrite existing file, if False, raise FileExistsError if file exists
        log: bool, default True
            if True, save ln(values), set nan for value <= 0
        color: bool, default True
            if True, save as color image
        showMask: bool, default True
            if True, show mask with green or nan, if False, mask is set as 0
        showCenter: bool, default False
            if True, draw center mark(tilted cross) with the same color as masked pixels
        """
        if (not overwrite) and (os.path.exists(path)):
            raise FileExistsError("")
        self.__buf = self.__raw.copy()
        self.__buf *= self.__masks
        if log:
            self.__log()
        if color:
            # shape=>(height,width,3), dtype=>uint8
            self.__toColor()
            if showMask:
                self.__buf[self.__masks == 0] = self.DEFAULT_MARK_COLOR
            if showCenter and self.__center[0] is not np.nan:
                self.__draw_center()
        cv2.imwrite(path, self.__buf)
        return 0

    def __log(self, nonPositiveValueAs=0) -> None:
        """update self.__buf with ln(self.__buf)

        Parameters
        ----------
        nonPositiveValueAs: 0 or np.nan, default 0
            value to set for non-positive value
        """
        if nonPositiveValueAs == 0:
            self.__buf = np.log(np.maximum(self.__buf, 1))
        elif nonPositiveValueAs == np.nan:
            self.__buf = np.log(self.__buf)
        else:
            raise ValueError("nonPositiveValueAs must be 0 or np.nan")
        return

    def __draw_center(
        self,
        *,
        color="Nan or green",
        markerType: int = cv2.MARKER_TILTED_CROSS,
        markerSize=100,
        thickness=2,
    ) -> None:
        """draw center mark on self.__buf

        Parameters
        ----------
        color: "Nan or green" or tuple of int, default "Nan or green"
            color of center mark, by default, green for colored image or nan for grayscale image
        markerType: int, default cv2.MARKER_TILTED_CROSS
            marker type of cv2.drawMarker
        markerSize: int, default 100
            marker size of cv2.drawMarker
        thickness: int, default 2
            thickness of cv2.drawMarker
        """
        if self.__center[0] is np.nan:
            return
        if color == "Nan or green":
            if len(self.__buf.shape) == 2:
                color = np.nan
            else:
                color = (0, 255, 0)
        center = (int(self.__center[0]), int(self.__center[1]))
        cv2.drawMarker(self.__buf, center, color, markerType, markerSize, thickness)
        return

    def __compress(self, dtype=np.uint8, *, min=None, max=None, setNanAs=None) -> None:
        """compress self.__buf to dtype

        Parameters
        ----------
        dtype: np.uint8 or np.uint16, default np.uint8
            dtype of compressed image
        min: numeric
            value to set as 0, by default, self.__buf.min()
        """
        if setNanAs is not None and np.any(self.__buf == np.nan):
            raise ValueError("self.__buf contains nan")

        if min is None:
            min = self.__buf.min()
        if max is None:
            max = self.__buf.max()

        zero2one: np.ndarray = (self.__buf - min) / (max - min)  # type: ignore
        if dtype == np.uint8:
            toCast = zero2one * ((1 << 8) - 1)
        elif dtype == np.uint16:
            toCast = zero2one * ((1 << 16) - 1)
        else:
            _logger.error(f"invalid dtype: {dtype}")
            raise ValueError("invalid dtype: only uint8 and uint16 are supported")
        toCast[toCast == np.nan] = setNanAs
        self.__buf = toCast.astype(dtype)
        return

    def __toColor(self, cmap=cv2.COLORMAP_HOT) -> None:
        """convert self.__buf grayscale array to color image

        Parameters
        ----------
        cmap: cv2.COLORMAP_*, default cv2.COLORMAP_HOT
        """
        self.__compress()
        self.__buf = cv2.applyColorMap(self.__buf, cmap)
        return

    def auto_mask_invalid(self, thresh: float = 0) -> None:
        """add mask for invalid pixels to self.__masks
        for data from pilatus sensor, negative values means invalid pixels
        """
        self.__masks.append(self.__raw >= thresh)  # nan=>0, otherwise=>1
        return

    def findcenter(self) -> tuple[float, float]:
        """find center and set to self.__center
        detect center by cv2.HoughCircles
        if no circle is detected, no error raised and self.__center is not updated

        Returns
        -------
        center: tuple[float,float]
            center of circle, (nan,nan) if no circle is detected
        """
        self.__buf = self.__raw.copy()
        cutoff = np.median(self.__buf)
        self.__buf[self.__buf < cutoff] = 0
        self.__compress(dtype=np.uint8)
        circles = cv2.HoughCircles(
            self.__buf,
            cv2.HOUGH_GRADIENT,
            1,
            20,
            param1=50,
            param2=30,
            minRadius=0,
            maxRadius=0,
        )
        if circles is None:
            _logger.info("no circle detected")
        else:
            for circle in circles[0, :]:
                _logger.debug(f"center: {circle[0]}, {circle[1]} radius: {circle[2]}")
            _logger.info(f"circle detected: {circles.shape}")
            self.__center = circles[0, 0, 0], circles[0, 0, 1]

        return self.center

    def integrate(
        self,
        *,
        dr: float = np.nan,
        bins: np.ndarray = np.arange(0),
        auto_findcenter: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """integrate along circumference

        Parameters
        ----------
        dr: float
            radius step, ignored if bins is specified
        bins: np.ndarray
            edges of each segment with length (number of segments)+1, if specified, dr is ignored
        range: tuple[float, float]
            range of radius, ignored if bins is specified

        Returns
        -------
        intensity: np.ndarray
            integrated intensity for each segment
        bins: np.ndarray
            bin edges finally used, same as input bins if specified
        """
        buf = self.__raw.copy()

        if self.center[0] is np.nan:
            if auto_findcenter:
                self.findcenter()
            else:
                raise UnboundLocalError("center not set yet")

        dx = np.ones_like(buf) * np.arange(buf.shape[1]) - self.__center[1]
        dy = (
            np.ones_like(buf) * np.arange(buf.shape[0]).reshape(-1, 1)
            - self.__center[0]
        )
        dist = np.sqrt(dx**2 + dy**2)
        buf = buf * 2 * np.pi * dist * self.__masks

        if bins.size == 0:
            if np.isnan(dr):
                raise ValueError("dr or bins must be specified")
            upper = np.nanmax(dist)
            num = int(np.floor(upper / dr))
            _logger.debug(f"upper: {upper}, num: {num}")
            bins = np.arange(num + 1) * dr

        intensity = np.zeros(bins.size - 1)
        for i in np.arange(bins.size - 1):
            bottom = bins[i]
            top = bins[i + 1]
            filter = (dist >= bottom) & (dist < top)
            num = np.sum(filter)
            sum = np.nansum(buf * filter)
            avg = sum / num
            intensity[i] = avg

        return intensity, bins

    @property
    def mask(self) -> np.ndarray:
        return np.asarray(self.__masks.copy())

    @mask.setter
    def mask(self, mask: np.ndarray) -> None:
        self.__masks.append(mask)


class Saxs1dProfile:
    INDEX = "index"
    PIXEL = "pixel"
    Qnm = "q[nm^-1]"
    X_UNITS = (INDEX, PIXEL, Qnm)

    def __init__(self, intensity: np.ndarray, bins: np.ndarray):
        if len(bins) != len(intensity):
            raise ValueError("invalid shape")
        self.__intensity: np.ndarray = intensity
        self.__pixels = bins
        self.__qnm: np.ndarray = None  # type: ignore
        self.__peaks: list[int] = None  # type: ignore
        return

    @property
    def intensity(self) -> np.ndarray:
        return self.__intensity.copy()

    @property
    def pixels(self) -> np.ndarray:
        return self.__pixels.copy()

    @property
    def q(self) -> np.ndarray:
        return self.__qnm.copy()

    def resetXUnit(self, x_unit: str, converter: Callable[[np.ndarray], np.ndarray]):
        """reset x unit and convert bins
        Parameters
        ----------
        x_unit: str
            new x unit
        func: function
            function to convert bins
        """
        self.__qnm = converter(self.__pixels)
        return

    def findpeaks(self, order: int = 1):
        """find peaks and set to self.peaks"""
        if order < 1:
            raise ValueError("order must be positive integer")
        n_rows = 2 * order + 1
        n_cols = len(self.__intensity) - n_rows
        table = np.ndarray((n_rows, n_cols))
        for i in range(n_rows):
            table[i, :] = self.__intensity[i : i + n_cols]
        peaks = np.argmax(table, axis=0) == order
        peak_idx = np.arange(n_cols)[peaks]
        self.__peaks = list(peak_idx)
        return

    def peaks(self, unit: str) -> list[int] | np.ndarray:
        """return nth peak"""
        if self.__peaks is None:
            self.findpeaks()

        left_idx = self.__peaks
        right_idx = [i + 1 for i in left_idx]
        if unit == "idx":
            return self.__peaks
        elif unit == self.PIXEL:
            ret = (self.__pixels[left_idx] + self.__pixels[right_idx]) / 2
            return ret  # type: ignore
        elif unit == self.Qnm:
            ret = (self.__qnm[left_idx] + self.__qnm[right_idx]) / 2
            return ret  # type: ignore
        else:
            raise ValueError("invalid unit")

    def save(
        self,
        path: str,
        *,
        overwrite: bool = False,
        fmt: str = "%.6e, %.6f",
        unit: str = "",
    ):
        """save profile as csv file"""
        if (not overwrite) and os.path.exists(path):
            raise FileExistsError("")
        if unit == "":
            unit = self.Qnm
        if unit == self.PIXEL:
            x = self.__pixels
        elif unit == self.Qnm:
            x = self.__qnm
        else:
            raise ValueError("invalid unit")
        header = f"bins[{unit}], intensity"
        np.savetxt(
            path,
            np.vstack((x, self.__intensity)).T,
            delimiter=",",
            header=header,
            fmt=fmt,
        )
        return


class SaxsProfile:
    def __new__(cls):
        raise NotImplementedError(f"{cls} default initializer not implemented")

    @classmethod
    def __internal_new(cls):
        return object.__new__(cls)

    def __init__(self):
        self.__img: Saxs2dProfile = None  # type: ignore
        self.__intensity: Saxs1dProfile = None  # type: ignore

    @classmethod
    def loadTiff(cls, path: str) -> "SaxsProfile":
        """load profile from tiff file

        Parameters
        ----------
        path: str
            path to tiff file

        Returns
        -------
        SaxsProfile
        """
        if not os.path.exists(path):
            raise FileNotFoundError("")
        if path[-4:] != ".tif":
            raise ValueError("invalid file type")
        obj = cls.__internal_new()
        obj.__img = Saxs2dProfile.load_tiff(path)
        return obj

    def integrate(self):
        """integrate and set to self.intensity"""
        self.__img.auto_mask_invalid()
        self.__img.findcenter()
        intensity, bins = self.__img.integrate(dr=1.0)
        pixels = (bins[:-1] + bins[1:]) / 2
        self.__intensity = Saxs1dProfile(intensity, pixels)
        return

    def convertToIq(
        self,
        converter: Callable[[np.ndarray], np.ndarray],
    ):
        if self.__intensity is None:
            raise UnboundLocalError("integrate not run yet")
        self.__intensity.resetXUnit(Saxs1dProfile.Qnm, converter)
        return

    @property
    def mask(self) -> np.ndarray:
        return self.__img.mask

    @mask.setter
    def mask(self, mask: np.ndarray) -> None:
        self.__img.mask = mask

    @property
    def center(self) -> tuple[float, float]:
        return self.__img.center

    @center.setter
    def center(self, center: tuple[float, float]):
        self.__img.center = center

    @property
    def intensity(self) -> np.ndarray:
        return self.__intensity.intensity

    def generateConverter(
        self, *, nthpeak: int, peak_q: float
    ) -> Callable[[np.ndarray], np.ndarray]:
        stdPixel = self.__intensity.peaks(Saxs1dProfile.PIXEL)[nthpeak - 1]
        return lambda x: x * peak_q / stdPixel

    def peaks(self, unit: str = "") -> list[int] | np.ndarray:
        return self.__intensity.peaks(unit)


def loadTiff(path: str) -> SaxsProfile:
    return SaxsProfile.loadTiff(path)
