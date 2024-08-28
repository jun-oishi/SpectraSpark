import numpy as np
import warnings, re, os
from SpectraSpark.util import listFiles, write_json, ArrayLike
from SpectraSpark.util.basic_calculation import r2q
from SpectraSpark.constants import PILATUS_PX_SIZE, DETECTER_PX_SIZES
from typing import Tuple
from numba import jit
import tqdm
import cv2

from ..util.io import savetxt

@jit(nopython=True, cache=True)
def _radial_average(img, center_x, center_y, threshold=2):
    """画像の中心を中心にして、動径平均を計算する

    Parameters
    ----------
    img : np.ndarray
        散乱強度の2次元配列
    center_x : int
        ビームセンターのx座標
    center_y : int
        ビームセンターのy座標
    threshold : float
        この値より小さい画素は無視する

    Returns
    -------
    r : np.ndarray
        動径[px]の配列
    i : np.ndarray
        動径方向の平均散乱強度の配列
    """
    width = img.shape[1]
    height = img.shape[0]
    r_mesh = np.empty(img.shape)
    dx_sq = (np.arange(width) - center_x)**2
    for y in range(height):
        r_mesh[y, :] = np.sqrt(dx_sq + (y - center_y)**2)

    min_r = int(r_mesh.min())
    max_r = int(r_mesh.max())+1
    r_range=max_r-min_r+1
    cnt = np.zeros(r_range, dtype=np.int64)
    i   = np.zeros(r_range, dtype=np.float64)
    for x in range(width):
        for y in range(height):
            if not img[y, x] >= threshold:
                continue
            idx = int(r_mesh[y, x]) - min_r
            cnt[idx] += 1
            i[idx] += img[y, x]

    for idx in range(len(cnt)):
        if cnt[idx] > 0:
            i[idx] /= cnt[idx]
        else:
            i[idx] = 0

    r = min_r + 0.5 + np.arange(len(cnt))
    return r, i

@jit(nopython=True, cache=True)
def _mask_and_average(img, mask, center_x, center_y, threshold=2):
    return _radial_average(img*mask, center_x, center_y, threshold)

def _readmask(src:str):
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} is not found.")
    mask = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if len(mask.shape) != 2:
        raise ValueError("mask file must be 2D single-channel image")
    mask[mask > 0] = 1
    return mask.astype(np.uint8)

def file_integrate(file:str, **kwargs):
    """SAXS画像を積分する

    see `saxs.series_integrate`
    """
    kwargs['verbose'] = False
    series_integrate(file, **kwargs)

def series_integrate(src: list[str]|str, *,
                     mask_src: str='', mask: np.ndarray=np.array([]),
                     center=(np.nan,np.nan),
                     camera_length=np.nan, wave_length=np.nan,
                     px_size=np.nan, detecter="",
                     slope=np.nan, intercept=np.nan,
                     flip='vertical',
                     dst="", overwrite=False, verbose=True):
    """SAXS画像の系列を積分する

    Parameters
    ----------
    dir : str
        画像ファイルのディレクトリ
    center : Tuple[float,float]
        ビームセンターの座標(x, y)
    camera_length : float
        カメラ長[mm]
    wave_length : float
        X線の波長[nm]
    px_size : float
        1pxのサイズ[mm]
    detecter : str
        検出器名(`PILATUS` or `EIGER`)
    slope : float
        線形回帰の傾き[nm^-1/px]
    intercept : float
        線形回帰の切片[nm^-1]
    flip : str
        ''なら反転無し、'v'なら上下反転、'h'なら左右反転、'vh'なら上下左右反転
    dst : str
        結果を保存するファイル名、指定がなければdir.csv
    overwrite : bool
        Trueなら上書きする
    verbose : bool
        Trueなら進捗バーを表示する
    """
    files:list[str] = []
    if isinstance(src, str):
        if src.endswith(".tif"):
            if not os.path.exists(src):
                raise FileNotFoundError(f"{src} is not found.")
            dst = dst if dst else re.sub(r"\.tif$", ".csv", src)
            files = [src]
        else:
            if not os.path.isdir(src):
                raise FileNotFoundError(f"{src} is not found.")
            files = [os.path.join(src, f) for f in listFiles(src, ext=".tif")]
            dst = dst if dst else src + ".csv"
    else:
        for file in src:
            if not file.endswith(".tif"):
                raise ValueError("Unsupported file format: only .tif is supported")
        if len(dst) == 0:
            raise ValueError("dst to save results must be set")
        files=src

    n_files = len(files)
    if n_files == 0:
        raise FileNotFoundError(f"No tif files in {src}.")

    if verbose:
        bar = tqdm.tqdm(total=n_files)

    if not overwrite and os.path.exists(dst):
        raise FileExistsError(f"{dst} is already exists.")

    i_all = []
    headers = ["q[nm^-1]"]

    if detecter.upper() in DETECTER_PX_SIZES:
        px_size = DETECTER_PX_SIZES[detecter.upper()]
    elif detecter == '':
        if np.isnan(px_size):
            raise ValueError("either `px_size` or `detecter` must be set")
    else:
        raise ValueError(f'unrecognized detecter `{detecter}`')

    calibration = 'none'
    if camera_length > 0 and wave_length > 0:
        calibration = 'geometry'
    elif not np.isnan(slope) and not np.isnan(intercept):
        calibration = 'linear_regression'
    else:
        warnings.warn("no valid calibration parameter given")

    height, width = cv2.imread(files[0], cv2.IMREAD_UNCHANGED).shape
    mask_flg = False
    if mask.size > 0:
        if mask.shape != (height, width):
            raise ValueError("mask size not match.")
        mask_flg = True
    else:
        if mask_src:
            mask = _readmask(mask_src)
            if mask.shape != (height, width):
                raise ValueError(f"mask size not match. {mask_src}")
            mask_flg = True

    r, i = np.array([]), np.array([])
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape != (height, width):
            raise ValueError(f"Image size is not match. {file}")
        if 'v' in flip:
            img = np.flipud(img)
        if 'h' in flip:
            img = np.fliplr(img)
        if mask_flg:
            r, i = _mask_and_average(img, mask, center[0], center[1])
        else:
            r, i = _radial_average(img, center[0], center[1])
        i_all.append(i)
        headers.append(os.path.basename(file))
        if verbose:
            bar.update(1)

    if calibration == 'geometry':
        q = r2q(r, camera_length, wave_length=wave_length, px_size=px_size)
    elif calibration == 'linear_regression':
        q = intercept + slope * r
    else:
        q = r * px_size
        headers[0] = "r[mm]"

    arr_out = np.hstack([q.reshape(-1, 1), np.array(i_all).T])
    savetxt(dst, arr_out, header=headers)

    paramfile = dst.replace(".csv", "_params.json")

    params={
        'center_x[px]': center[0],
        'center_y[px]': center[1],
        'calibration_type': calibration,
        'px_size[mm]': px_size,
        'camera_length[mm]': camera_length,
        'wave_length[AA]': wave_length,
        'slope[nm^-1/px]': slope,
        'intercept[nm^-1]': intercept,
    }
    if 'v' in flip and 'h' in flip:
        flip = 'vertical and horizontal'
    elif 'v' in flip:
        flip = 'vertical'
    elif 'h' in flip:
        flip = 'horizontal'
    else:
        flip = 'none'
    params['flip'] = flip
    write_json(paramfile, params)

    if verbose:
        bar.close()
    return dst

class Mask:
    """値が0の画素を無視するマスク"""
    def __init__(self, shape=(0,0), value:np.ndarray|None=None):
        if value is not None:
            self.__mask = value.astype(np.uint8)
        else:
            if shape[0] <= 0 or shape[1] <= 0:
                raise ValueError("Invalid shape")
            self.__mask = np.ones(shape, dtype=np.uint8)
        return

    @property
    def value(self, dtype=np.uint8) -> np.ndarray:
        return (self.__mask > 0).astype(dtype)

    def apply(self, arr:np.ndarray):
        return arr * (self.value>0).astype(arr.dtype)

    @property
    def shape(self):
        return self.__mask.shape

    def add(self, arr:np.ndarray):
        self.__mask[arr > 0] = 0
        return

    def add_rectangle(self, x: int, y: int, width: int, height: int):
        self.__mask[y:y+height, x:x+width] = 0
        return

    def remove_rectangle(self, x: int, y: int, width: int, height: int):
        self.__mask[y:y+height, x:x+width] = 1
        return

    def save(self, file: str='mask.pbm'):
        cv2.imwrite(file, self.__mask)
        return

    @classmethod
    def read(cls, src: str):
        return cls(value=_readmask(src))

class Saxs2d:
    def __init__(self, i: np.ndarray, px2q: float, center: ArrayLike):
        self.__i = i  # floatの2次元配列 欠損値はnp.nan
        self.__px2q = px2q  # nm^-1/px
        self.__center = (center[0], center[1])  # (x,y)
        return

    @property
    def i(self) -> np.ndarray:
        return self.__i

    @property
    def center(self) -> Tuple[float, float]:
        return self.__center

    @property
    def px2q(self) -> float:
        return self.__px2q

    def radial_average(
        self, q_min: float = 0, q_max: float = np.inf
    ) -> Tuple[np.ndarray, np.ndarray]:
        rx = np.arange(self.__i.shape[1]) - self.__center[0]
        ry = np.arange(self.__i.shape[0]) - self.__center[1]
        rxx, ryy = np.meshgrid(rx, ry)
        r = np.sqrt(rxx**2 + ryy**2)  # type: ignore

        r_min = int(np.floor(q_min / self.__px2q))
        r_max = int(np.ceil(min(q_max / self.__px2q, r.max())))
        r_bin = np.arange(r_min, r_max + 1, 1)

        r[np.isnan(self.__i)] = np.nan
        cnt = np.histogram(r, bins=r_bin)[0]
        i_sum = np.histogram(r, bins=r_bin, weights=self.__i)[0]
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        i = i_sum / cnt
        warnings.resetwarnings()

        q_bin = r_bin * self.__px2q
        return i, (q_bin[:-1] + q_bin[1:]) / 2

    def rotate(self, angle: float):
        """画像を回転する"""
        raise NotImplementedError
