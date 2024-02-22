import numpy as np
from numba import njit

# x = a*[1,0], y = a*[-1/2, sqrt(3)/2]の三角格子状の粒子配置を扱う

# jitコンパイルされた関数内ではこれらの変数はコンパイル時の値に固定されて読み取り専用として使える
_SQRT3BY2 = np.sqrt(3) / 2
_N_THETA = 360  # 方位平均を取るための角度の分割数
_DTHETA = 2 * np.pi / _N_THETA  # 方位平均を取るための角度の分割幅
_STEP = (
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
)  # 6方向への移動のためのベクトル

_EMPTY = np.empty((0))


@njit(cache=True)
def _shuffled_range(n: int) -> np.ndarray:
    """0からn-1までの整数をランダムに並べた配列を返す"""
    # ref. fisher-yates shuffle
    ret = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        ret[i], ret[j] = ret[j], ret[i]
    return ret


def _uint(val) -> int:
    if val % 1 > 1e-6 or val < 0:
        raise ValueError("val must be a natural number")
    return int(val)


@njit(cache=True)
def _real_coords(x: np.ndarray, y: np.ndarray, a: float) -> np.ndarray:
    """格子座標の配列を実座標に変換する

    Args:
        x (np.ndarray): 格子座標のx座標[int]の配列
        y (np.ndarray): 格子座標のy座標[int]の配列
        a (float): 格子定数

    Returns:
        np.ndarray: 1列目がx座標、2列目がy座標の実座標の配列
    """
    ret = np.empty((x.size, 2), dtype=np.float64)
    ret[:, 0] = x * a - y * a / 2
    ret[:, 1] = y * a * _SQRT3BY2
    return ret


@njit(cache=True)
def _real_coord(x: int, y: int, a: float) -> tuple[float, float]:
    """格子座標を実座標に変換する

    Args:
        x (int): 格子座標のx座標
        y (int): 格子座標のy座標
        a (float): 格子定数

    Returns:
        tuple[float, float]: (x座標, y座標)
    """
    return x * a - y * a / 2, y * a * _SQRT3BY2


@njit(cache=True)
def _initial_arrange(
    Lx: int, Ly: int, n: int, cutoff: float, a: float
) -> tuple[np.ndarray, np.ndarray]:
    """粒子を初期配置する

    Args:
        Lx (int): もでる領域のx方向のサイズ
        Ly (int): モデル領域のy方向のサイズ
        n (int): 粒子数
        cutoff (float): カットオフ距離
        a (float): 格子定数

    Raises:
        RuntimeError: カットオフ距離内に粒子がないように配置できなかった場合

    Returns:
        tuple[np.ndarray, np.ndarray]: 格子座標のx座標の配列, y座標の配列
    """
    x = np.empty((n), dtype=np.int16)
    y = np.empty((n), dtype=np.int16)
    for i in range(n):
        fail = True
        for _x in _shuffled_range(Lx):
            x[i] = _x

            for _y in _shuffled_range(Ly):
                y[i] = _y
                in_cutoff = False
                # print("putting", i, "at", x[i], y[i])
                for j in range(i):
                    if _distance(x[i], y[i], x[j], y[j], Lx, Ly, a) < cutoff:
                        # print("  ", j, "at", x[j], y[j], "in cutoff")
                        in_cutoff = True
                        break
                if in_cutoff:
                    continue
                fail = False
                break

            if fail:
                continue
            break

        if fail:
            raise RuntimeError("Failed to arrange particles")
    return x, y


@njit(cache=True)
def _distance(x1: int, y1: int, x2: int, y2: int, Lx: int, Ly: int, a: float) -> float:
    """2点間の距離を計算する

    Args:
        x1 (int): 1点目のx座標(格子座標)
        y1 (int): 1点目のy座標(格子座標)
        x2 (int): 2点目のx座標(格子座標)
        y2 (int): 2点目のy座標(格子座標)
        Lx (int): モデル空間のx方向のサイズ
        Ly (int): モデル空間のy方向のサイズ
        a (float): 格子定数

    Returns:
        float: 周期的境界条件を考慮した2点間の距離
    """
    x = np.array([x2 - Lx, x2, x2 + Lx, x2 - Lx, x2, x2 + Lx, x2 - Lx, x2, x2 + Lx])
    y = np.array([y2 - Ly, y2 - Ly, y2 - Ly, y2, y2, y2, y2 + Ly, y2 + Ly, y2 + Ly])
    xy2 = _real_coords(x, y, a)
    xy1 = _real_coord(x1, y1, a)
    d = np.sqrt((xy2[:, 0] - xy1[0]) ** 2 + (xy2[:, 1] - xy1[1]) ** 2)
    return d.min()


@njit(cache=True)
def _compute_i_par(q: np.ndarray, r_par: float) -> np.ndarray:
    """1粒子の散乱因子を求める

    Args:
        q (np.ndarray): 散乱ベクトルの大きさの配列
        r_par (float): 粒子半径

    Returns:
        np.ndarray: qに対応する散乱因子の配列
    """
    # ref.Matsuoka, Nihon Kessho Gakkaishi 1999
    x = q * r_par
    return (3 * (np.sin(x) - x * np.cos(x)) / x**3) ** 2


@njit(cache=True)
def _compute_a(q: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """散乱振幅を計算する
    戻り値は二次元配列2つのタプルでそれぞれの配列の形状は(q.size, _N_THETA)で
    re[i,j]=<q[i]が向きjで入るときの全粒子の複素散乱振幅exp(-iqr)の和の実部>
    im[i,j]=<同虚部>

    Args:
        q (np.ndarray): qの配列
        xy (np.ndarray): 1列目がx座標、2列目がy座標の実座標の配列

    Returns:
        tuple[np.ndarray, np.ndarray]: 散乱振幅の実部と虚部の配列
    """
    _re = np.empty((q.size, _N_THETA), dtype=np.float64)
    _im = np.empty((q.size, _N_THETA), dtype=np.float64)
    # この書き方ならxyの形が違うとエラーになる and 多分アドレス渡しされるので効率的
    x, y = xy.T
    for i in range(q.size):
        for j in range(_N_THETA):
            theta = j * _DTHETA
            qx, qy = q[i] * np.cos(theta), q[i] * np.sin(theta)
            qr = qx * x + qy * y
            _re[i, j] = np.cos(-qr).sum()
            _im[i, j] = np.sin(-qr).sum()
    return _re, _im


@njit(cache=True)
def _a2i(re: np.ndarray, im: np.ndarray) -> np.ndarray:
    """散乱振幅の実部と虚部から散乱強度(構造因子)を計算する"""
    i = np.sum(re**2 + im**2, axis=1) * _DTHETA / (2 * np.pi)
    return i


@njit(cache=True)
def _residual(i_exp: np.ndarray, i_calc: np.ndarray) -> float:
    """散乱強度の実験値とモデルからの計算値の残差を計算して返す
    実験値と計算値は総和が1になるように規格化してから計算する

    Args:
        i_exp (np.ndarray): 実験データの散乱強度の配列(規格化不要)
        i_calc (np.ndarray): 計算された散乱強度の配列(規格化不要)

    Returns:
        float: 残差 和が1の配列の二乗誤差の総和なので1くらいの値のはず
    """
    return np.sum((i_exp / i_exp.sum() - i_calc / i_calc.sum()) ** 2)


@njit(cache=True)
def _step_forword(
    x: np.ndarray,
    y: np.ndarray,
    Lx: int,
    Ly: int,
    a: float,
    cutoff: float,
    q: np.ndarray,
    i_exp: np.ndarray,
    i_par: np.ndarray,
    last_res: float,
    a_re: np.ndarray,
    a_im: np.ndarray,
    sigma2: float,
    n_moves: int = 3,
) -> float:
    """1ステップ進めてx, y, a_re, a_im を破壊的に更新して残差を返す

    Args:
        x (np.ndarray): 格子座標のx座標の配列
        y (np.ndarray): 格子座標のy座標の配列
        Lx (int): モデル領域のx方向のサイズ
        Ly (int): モデル領域のy方向のサイズ
        a (float): 格子定数
        cutoff (float): カットオフ距離
        q (np.ndarray): 散乱ベクトルの配列
        i_exp (np.ndarray): 実験データの散乱強度の配列
        i_par (np.ndarray): 1粒子の散乱因子の配列
        last_res (float): 前ステップのres
        a_re (np.ndarray): 散乱振幅の実部の配列
        a_im (np.ndarray): 散乱振幅の虚部の配列

    Returns:
        float: 更新後のres
    """
    n = x.size

    # ランダムに粒子を選び、移動する
    n_moves_remained = n_moves
    old_x, old_y = x.copy(), y.copy()
    for i in _shuffled_range(n):
        for d in _shuffled_range(6):
            move = _STEP[d]
            new_x = (x[i] + move[0]) % Lx
            new_y = (y[i] + move[1]) % Ly

            in_cutoff = False
            # print("trying to move", i, "to", new_x, new_y)
            for j in range(n):
                if i == j:
                    continue
                if _distance(new_x, new_y, x[j], y[j], Lx, Ly, a) < cutoff:
                    # print("  ", j, "at", x[j], y[j], "in cutoff")
                    in_cutoff = True
                    break
            if in_cutoff:
                continue

            # 座標と散乱振幅を更新する
            n_moves_remained -= 1
            x[i], y[i] = new_x, new_y
            # 移動による差分を使って散乱振幅を更新する[粒子数に対してO(0)で高速]
            xy_old = _real_coord(old_x[i], old_y[i], a)
            xy_new = _real_coord(x[i], y[i], a)

            _new_re, _new_im = a_re.copy(), a_im.copy()
            for i in range(q.size):
                for j in range(_N_THETA):
                    theta = j * _DTHETA
                    qx, qy = q[i] * np.cos(theta), q[i] * np.sin(theta)
                    qr_old = qx * xy_old[0] + qy * xy_old[1]
                    qr_new = qx * xy_new[0] + qy * xy_new[1]
                    _new_re[i, j] += np.cos(-qr_new) - np.cos(-qr_old)
                    _new_im[i, j] += np.sin(-qr_new) - np.sin(-qr_old)
            break

        if n_moves_remained <= 0:
            break

    if n_moves_remained > 0:
        raise RuntimeError("Failed to move particle")

    # 更新後の散乱強度を計算して更新の可否を判定する
    # ref. Guinier, 1955
    i_calc = i_par * _a2i(_new_re, _new_im)
    # ref. McGreevy, 2001
    new_res: float = _residual(i_exp, i_calc)

    # print("a_re:", a_re)
    # print("  -> ", _new_re)
    # print("i_calc:", i_calc)
    # print("i_exp :", i_exp)
    # print(f"res: {last_res:.3e} -> {new_res:.3e}", end=" ")

    if new_res < last_res or np.random.rand() < np.exp(
        -(new_res - last_res) / (2 * sigma2)
    ):
        # 添え字無しの代入だとアドレスが書き換えられて呼び出し元で更新されない
        a_re[:] = _new_re
        a_im[:] = _new_im
        # print("accepted")
        pass
    else:
        x[:] = old_x
        y[:] = old_y
        new_res = last_res
        # print("rejected")

    # print("x:", x, "y:", y)

    return new_res


@njit(cache=True)
def _run(
    x: np.ndarray,
    y: np.ndarray,
    Lx: int,
    Ly: int,
    a: float,
    cutoff: float,
    q: np.ndarray,
    i_exp: np.ndarray,
    r_par: float,
    max_iter: int,
    res_thresh: float,
    log_interval: int,
    thresh_interval: int,
    sigma2: float = 1,
    n_moves_in_step: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """フィッティングを実行する
    モンテカルロ法でmax_iterステップを上限としてthreshold_intervalステップ連続でresが閾値を下回ったら終了する
    sigma2はメトロポリス法のパラメータで値が大きいほどresが増加する移動を受け入れやすくなる
    log_intervalは全粒子の配置を記録するステップ間隔でresは毎ステップ記録される

    Args:
        x (np.ndarray): 格子座標のx座標の配列
        y (np.ndarray): 格子座標のy座標の配列
        Lx (int): モデル領域のx方向のサイズ
        Ly (int): モデル領域のy方向のサイズ
        a (float): 格子定数
        cutoff (float): カットオフ距離
        q (np.ndarray): 実験データの散乱ベクトルの配列
        i_exp (np.ndarray): 実験データの散乱強度の配列
        r_par (float): 粒子の半径
        max_iter (int): 最大ステップ数
        res_thresh (float): 残差の閾値
        log_interval (int): ログを取るステップ間隔
        thresh_interval (int): 終了条件のステップ数
        sigma2 (float, optional): メトロポリス法のパラメータ. Defaults to 1.

    Returns:
        res_log(np.ndarray): 残差の履歴
        logged_steps(np.ndarray): x_log, y_logの各行が記録されたステップ
        x_log(np.ndarray): xの履歴
        y_log(np.ndarray): yの履歴
    """
    log_len = int(max_iter / log_interval) + 2
    res_log = np.empty((max_iter + 1), dtype=np.float64)
    logged_steps = np.empty((log_len), dtype=np.uint32)
    x_log = np.empty((log_len, x.size), dtype=np.int16)
    y_log = np.empty_like(x_log)

    i_par = _compute_i_par(q, r_par)
    a_re, a_im = _compute_a(q, _real_coords(x, y, a))
    i_calc = i_par * _a2i(a_re, a_im)
    res: float = _residual(i_exp, i_calc)
    res_log[0] = res
    logged_steps[0] = 0
    x_log[0] = x
    y_log[0] = y

    n_iter = 0
    thresh_count = 0
    log_idx = 1
    n_iter = 1
    converged = False
    while n_iter <= max_iter:
        res = _step_forword(
            x=x,
            y=y,
            Lx=Lx,
            Ly=Ly,
            a=a,
            cutoff=cutoff,
            q=q,
            i_exp=i_exp,
            i_par=i_par,
            last_res=res,
            a_re=a_re,
            a_im=a_im,
            sigma2=sigma2,
            n_moves=n_moves_in_step,
        )  # x, y, a_re, a_im は破壊的に更新される
        res_log[n_iter] = res
        n_iter += 1

        if res < res_thresh:
            thresh_count += 1
            # print(f"thresh_count: {thresh_count}")
            if thresh_count >= thresh_interval:
                # 最後のステップを記録して終了
                logged_steps[log_idx] = n_iter
                x_log[log_idx] = x
                y_log[log_idx] = y
                converged = True
                log_idx += 1
                break
        else:
            thresh_count = 0
            # print(f"thresh_count: {thresh_count}")

        if n_iter % log_interval == 0:
            logged_steps[log_idx] = n_iter
            x_log[log_idx] = x
            y_log[log_idx] = y
            log_idx += 1

        # 誤差が蓄積するようなのでたまにO(N)で再計算
        if n_iter % thresh_interval == 0:
            a_re, a_im = _compute_a(q, _real_coords(x, y, a))

        # input("press enter to continue")

        continue

    res_log = res_log[:n_iter]
    logged_steps = logged_steps[:log_idx]
    x_log = x_log[:log_idx]
    y_log = y_log[:log_idx]
    return res_log, logged_steps, x_log, y_log, converged


class TriangleSimulator:
    def __init__(
        self,
        *,
        Lx: int = -1,
        Ly: int = -1,
        n: int = -1,
        cutoff: float = 0.650,
        a: float = 0.321,
        r_par: float = 0.355,
    ):
        """三角格子上の粒子の配置をシミュレーションするクラス

        Args:
            Lx (int): モデル領域のx方向のサイズ
            Ly (int): モデル領域のy方向のサイズ
            n (int): 粒子数
            cutoff (float, optional): 許容される粒子の最近接の距離. Defaults to 0.650.
            a (float, optional): 格子定数. Defaults to 0.321.
            r_par (float, optional): 粒子の半径. Defaults to 0.355.
        """
        self.__Lx: int = Lx  # 2次元格子のサイズ
        self.__Ly: int = Ly if Ly > 0 else Lx  # 2次元格子のサイズ
        self.__n: int = n  # 粒子数
        self.__cutoff: float = cutoff  # 最接近粒子間の距離
        self.__a: float = a  # 格子定数
        self.__r_par: float = r_par  # 散乱計算に使う粒子半径

        if self.Lx > 0 and self.Ly > 0 and self.n > 0 and self.a > 0:
            self.__x, self.__y = _initial_arrange(
                self.Lx, self.Ly, self.n, self.cutoff, self.a
            )

        self.__q = np.array([])  # 散乱ベクトル
        self.__max_iter: int = -1
        self.__res_thresh: float = np.nan
        self.__sigma2: float = np.nan
        self.__log_interval: int = -1
        self.__thresh_interval: int = -1
        self.__i_exp = np.array([])
        self.__res_log = np.array([])
        self.__logged_steps = np.array([])
        self.__x_log = np.array([])
        self.__y_log = np.array([])
        self.__q_min: float = 0.0
        self.__q_max: float = np.inf
        return

    @classmethod
    def load(cls, src: str) -> "TriangleSimulator":
        raise NotImplementedError

    @property
    def Lx(self) -> int:
        """モデル領域のx方向のサイズ[格子単位]"""
        return self.__Lx

    @Lx.setter
    def Lx(self, value: int):
        self.__Lx = _uint(value)
        if self.Ly < 0:
            self.__Ly = self.Lx
        if self.n > 0 and self.a > 0:
            self.__x, self.__y = _initial_arrange(
                self.Lx, self.Ly, self.n, self.cutoff, self.a
            )

    @property
    def Ly(self) -> int:
        """モデル領域のy方向のサイズ[格子単位]"""
        return self.__Ly

    @Ly.setter
    def Ly(self, value: int):
        self.__Ly = _uint(value)
        if self.Lx < 0:
            self.__Lx = self.Ly
        if self.n > 0 and self.a > 0:
            self.__x, self.__y = _initial_arrange(
                self.Lx, self.Ly, self.n, self.cutoff, self.a
            )

    @property
    def n(self) -> int:
        """粒子数"""
        return self.__n

    @n.setter
    def n(self, value: int):
        self.__n = _uint(value)
        if self.Lx > 0 and self.Ly > 0 and self.a > 0:
            self.__x, self.__y = _initial_arrange(
                self.Lx, self.Ly, self.n, self.cutoff, self.a
            )

    @property
    def x(self) -> np.ndarray:
        """格子座標のx座標の配列"""
        return self.__x

    @property
    def y(self) -> np.ndarray:
        """格子座標のy座標の配列"""
        return self.__y

    @property
    def cutoff(self) -> float:
        """許容する最近接粒子間の距離[nm]"""
        return self.__cutoff

    @cutoff.setter
    def cutoff(self, value: float):
        self.__cutoff = float(value)

    @property
    def a(self) -> float:
        """格子定数[nm]"""
        return self.__a

    @a.setter
    def a(self, value: float):
        self.__a = float(value)

    @property
    def r_par(self) -> float:
        """粒子の半径[nm]"""
        return self.__r_par

    @r_par.setter
    def r_par(self, value: float):
        self.__r_par = float(value)

    @property
    def q(self) -> np.ndarray:
        """散乱ベクトルの配列[nm^-1]"""
        return self.__q

    def _set_q(self, q: np.ndarray):
        self.__q = q

    @property
    def i_exp(self) -> np.ndarray:
        """実験データの散乱強度の配列 和が1になるように規格化されている"""
        return self.__i_exp / self.__i_exp.sum()

    @property
    def max_iter(self) -> int:
        """モンテカルロ法の最大ステップ数"""
        return self.__max_iter

    @max_iter.setter
    def max_iter(self, value: int):
        self.__max_iter = _uint(value)

    @property
    def res_thresh(self) -> float:
        """残差の閾値"""
        return self.__res_thresh

    @res_thresh.setter
    def res_thresh(self, value: float):
        self.__res_thresh = float(value)

    @property
    def sigma2(self) -> float:
        """メトロポリス法のパラメータ"""
        return self.__sigma2

    @sigma2.setter
    def sigma2(self, value: float):
        self.__sigma2 = float(value)

    @property
    def log_interval(self) -> int:
        """全粒子の位置を記録するステップ間隔"""
        return self.__log_interval

    @log_interval.setter
    def log_interval(self, value: int):
        self.__log_interval = _uint(value)
        if self.thresh_interval < 0:
            self.__thresh_interval = self.log_interval

    @property
    def thresh_interval(self) -> int:
        """残差が閾値を下回ってから終了するまでのステップ数"""
        return self.__thresh_interval

    @thresh_interval.setter
    def thresh_interval(self, value: int):
        self.__thresh_interval = _uint(value)
        if self.log_interval < 0:
            self.__log_interval = self.thresh_interval

    @property
    def res_log(self) -> np.ndarray:
        """残差の履歴"""
        return self.__res_log

    @property
    def logged_steps(self) -> np.ndarray:
        """全粒子の位置を記録したステップの配列"""
        return self.__logged_steps

    @property
    def x_log(self) -> np.ndarray:
        """全粒子のx座標の履歴"""
        return self.__x_log

    @property
    def y_log(self) -> np.ndarray:
        """全粒子のy座標の履歴"""
        return self.__y_log

    @property
    def converged(self) -> bool:
        """フィッティングが収束したかどうか"""
        return self.__converged

    @property
    def q_range(self) -> tuple[float, float]:
        """評価するqの範囲"""
        return self.__q_min, self.__q_max

    @q_range.setter
    def q_range(self, q_range: tuple[float, float]):
        self.__q_min = float(q_range[0])
        self.__q_max = float(q_range[1])
        mask = (self.q >= q_range[0]) & (self.q <= q_range[1])
        self.__q = self.q[mask]
        self.__i_exp = self.i_exp[mask]

    def load_exp_data(
        self,
        src: str,
        *,
        q_range: tuple[float, float] = (0.0, np.inf),
        delimiter: str = ",",
    ):
        """実験データを読み込む
        フォーマットはcsvで、"#"で始まる行はコメントとして無視される

        Args:
            src (str): ファイルのパス
            q_range (tuple[float, float], optional): フィッティングするqの範囲. Defaults to (0.0, np.inf).
            delimiter (str, optional): 区切り文字. Defaults to ",".
        """
        comments = "#"
        data = np.loadtxt(src, delimiter=delimiter, comments=comments, dtype=np.float64)
        self.__q = data[:, 0]
        self.__i_exp = data[:, 1]
        self.q_range = q_range
        return

    @property
    def real_coords(self) -> np.ndarray:
        """全粒子の実座標の配列"""
        return _real_coords(self.x, self.y, self.a)

    def _distance(self, i: int, j: int) -> float:
        """i番目の粒子とj番目の粒子の距離を計算する"""
        return _distance(
            self.x[i], self.y[i], self.x[j], self.y[j], self.Lx, self.Ly, self.a
        )

    def compute_i(self, x: np.ndarray = _EMPTY, y: np.ndarray = _EMPTY) -> np.ndarray:
        """散乱強度を計算する"""
        if x.size == 0 and y.size == 0:
            x, y = self.x, self.y
        i_par = _compute_i_par(self.q, self.r_par)
        _re, _im = _compute_a(self.q, _real_coords(x, y, self.a))
        return i_par * _a2i(_re, _im)

    def run(
        self,
        *,
        max_iter: int = -1,
        res_thresh: float = np.nan,
        log_interval: int = -1,
        thresh_interval: int = -1,
        sigma2: float = np.nan,
        n_moves_in_step: int = 3,
    ) -> bool:
        """フィッティングを実行する

        Args:
            max_iter (int): 最大のステップ数
            res_thresh (float): 残差の閾値
            log_interval (int): ログを取るステップ間隔
            thresh_interval (int, optional): 残差が閾値を下回ってから収束と判定するまでのステップ数. Defaults to log_interval.

        Returns:
            bool: フィッティングが収束したらTrue
        """
        if thresh_interval < 0:
            thresh_interval = log_interval
        self.max_iter = max_iter if max_iter > 0 else self.max_iter
        self.res_thresh = res_thresh if not np.isnan(res_thresh) else self.res_thresh
        self.thresh_interval = (
            thresh_interval if thresh_interval > 0 else self.thresh_interval
        )
        self.log_interval = log_interval if log_interval > 0 else self.log_interval
        self.sigma2 = sigma2 if not np.isnan(sigma2) else self.sigma2

        if self.max_iter < 0:
            raise ValueError("max_iter is not set")
        if np.isnan(self.res_thresh):
            raise ValueError("res_thresh is not set")
        if self.log_interval < 0:
            raise ValueError("log_interval is not set")
        if self.thresh_interval < 0:
            raise ValueError("thresh_interval is not set")
        if np.isnan(self.sigma2):
            raise ValueError("sigma2 is not set")

        res_log, logged_steps, x_log, y_log, converged = _run(
            x=self.x,
            y=self.y,
            Lx=self.Lx,
            Ly=self.Ly,
            a=self.a,
            cutoff=self.cutoff,
            q=self.q,
            i_exp=self.i_exp,
            r_par=self.r_par,
            max_iter=self.max_iter,
            res_thresh=self.res_thresh,
            log_interval=self.log_interval,
            thresh_interval=self.thresh_interval,
            sigma2=self.sigma2,
            n_moves_in_step=n_moves_in_step,
        )
        self.__res_log = res_log
        self.__logged_steps = logged_steps
        self.__x_log = x_log
        self.__y_log = y_log
        self.__x = x_log[-1, :]
        self.__y = y_log[-1, :]
        self.__converged = converged

        return self.converged

    def load_arrangement(self, src: str, comments="#", axis=1, check_n=False):
        """粒子の配置を読み込む
        commentsから始まる行はコメントとして無視される
        axis=1なら1行目をx, 2行目をyとして、axis=0なら1列目をx, 2列目をyとして読み込む
        """
        table = np.loadtxt(src, comments=comments, dtype=np.int16)
        if axis == 0:
            table = table.T

        if check_n and table.shape[1] != self.n:
            raise ValueError("The number of particles does not match")
        self.__x = table[0, :]
        self.__y = table[1, :]
        self.__y = self.Ly - self.y
        self.__n = self.x.size

    def save_result(self, name: str):
        """結果を保存する
        与えられた名前に拡張子をつけて保存する

        Args:
            name (str): 保存するファイルの名前
        """
        # config
        dst = name + ".conf"
        with open(dst, "w") as f:
            f.write(f"Lx = {self.Lx}a\n")
            f.write(f"Ly = {self.Ly}a\n")
            f.write(f"n = {self.n}\n")
            f.write(f"cutoff = {self.cutoff} [nm]\n")
            f.write(f"a = {self.a} [nm]\n")
            f.write(f"r_par = {self.r_par} [nm]\n")
            f.write(f"q_range = {self.q_range} [nm^-1]\n")
            f.write(f"max_iter = {self.max_iter} steps\n")
            f.write(f"res_thresh = {self.res_thresh}\n")
            f.write(f"log_interval = {self.log_interval}\n")
            f.write(f"thresh_interval = {self.thresh_interval}\n")
            f.write(f"converged = {self.converged}\n")
            f.write(f"\n-----------------------\n\n")
            f.write("exp data to fit\n")
            f.write("q[nm^-1]  i_exp\n")
            for q, i in zip(self.q, self.i_exp):
                f.write(f"{q} {i}\n")

        # res log
        dst = name + ".res_log"
        with open(dst, "w") as f:
            f.write(f"# step  res\n")
            for i, res in enumerate(self.res_log):
                f.write(f"{i:>6}  {res}\n")

        # arrangement history
        fmt = "%d"
        dst = name + ".x_hist"
        header = "step " + "\t".join([f"x{i:03}" for i in range(self.n)])
        np.savetxt(
            dst,
            np.column_stack([self.logged_steps, self.x_log]),
            delimiter="\t",
            header=header,
            fmt=fmt,
        )
        dst = name + ".y_hist"
        header = "step " + "\t".join([f"y{i:03}" for i in range(self.n)])
        np.savetxt(
            dst,
            np.column_stack([self.logged_steps, self.y_log]),
            delimiter="\t",
            header=header,
            fmt=fmt,
        )

        return
