import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import overload

from .rmc import TriangleSimulator

_SQRT3BY2 = np.sqrt(3) * 0.5

# 第N近接原子の距離の係数(1NNはじまり)
_NEIGHBOR_FACTOR = (
    1,
    np.sqrt(3),
    2,
    np.sqrt(7),
    3,
    np.sqrt(12),
    np.sqrt(13),
    4,
    np.sqrt(19),
    np.sqrt(21),
    5,
)


def _draw_lattice(
    ax: Axes, Lx: int, Ly: int, a: float, color: str = "gray", linewidth: float = 0.5
) -> Axes:
    """三角格子を描画する

    Args:
        ax (Axes): 描画するAxes
        Lx (int): モデル領域のx方向の格子数
        Ly (int): モデル領域のy方向の格子数
        a (float): 格子定数
        color (str, optional): 線色. Defaults to "gray".
        linewidth (float, optional): 線幅. Defaults to 0.5.

    Returns:
        Axes: 描画されたAxes
    """
    ax.set_aspect("equal")
    for i in range(Lx + 1):
        ini = (i * a, 0)
        fin = (i * a - Ly * a / 2, Ly * a * np.sqrt(3) / 2)
        ax.plot([ini[0], fin[0]], [ini[1], fin[1]], color=color, linewidth=linewidth)
    for i in range(Ly + 1):
        ini = (-i * a / 2, i * a * np.sqrt(3) / 2)
        fin = (Lx * a - i * a / 2, i * a * np.sqrt(3) / 2)
        ax.plot([ini[0], fin[0]], [ini[1], fin[1]], color=color, linewidth=linewidth)

    for i in range(Lx + Ly):
        if i < Ly:
            ini = ((-Ly + i) * a * 0.5, (Ly - i) * a * _SQRT3BY2)
        else:
            ini = ((i - Ly) * a, 0)
        if i <= Lx:
            fin = (-Ly * a * 0.5 + i * a, Ly * a * _SQRT3BY2)
        else:
            fin = (
                (-Ly * 0.5 + Lx) * a + (i - Lx) * a * 0.5,
                (Ly - (i - Lx)) * a * _SQRT3BY2,
            )
        ax.plot([ini[0], fin[0]], [ini[1], fin[1]], color=color, linewidth=linewidth)

    ax.set_xlabel(r"$x$ [nm]")
    ax.set_ylabel(r"$y$ [nm]")
    return ax


def _detect_neighbors(
    sim: TriangleSimulator, max_nn: int, *, exact=False
) -> list[list[list[int]]]:
    """近接リストを生成する
    exactがTrueなら周期的境界条件を考慮する(未実装)

    Args:
        sim (TriangleSimulator): 粒子配置を持つシミュレータ
        max_nn (int): 検出する最大距離(1NN, 2NN, ...)
        exact (bool, optional): Trueなら周期的境界条件を考慮する. Defaults to False.

    Returns:
        list[list[list[int]]]: ret[i][k]はi番目の粒子の(k+1)NNのリスト
    """
    xy = sim.real_coords
    neighbors = [[[] for k in range(max_nn + 1)] for i in range(sim.n)]

    borders = sim.a * np.array(
        [(_NEIGHBOR_FACTOR[k] + _NEIGHBOR_FACTOR[k + 1]) / 2 for k in range(max_nn)]
    )

    a = sim.Lx * sim.a * np.array([1.0, 0.0])
    b = sim.Lx * sim.a * np.array([-0.5, _SQRT3BY2])
    for i in range(sim.n):
        for j in range(sim.n):
            d = np.linalg.norm(xy[i] - xy[j])
            if exact:
                d = min(
                    np.linalg.norm(xy[i] - xy[j] - a - b),
                    np.linalg.norm(xy[i] - xy[j] - a),
                    np.linalg.norm(xy[i] - xy[j] - a + b),
                    np.linalg.norm(xy[i] - xy[j] - b),
                    np.linalg.norm(xy[i] - xy[j]),
                    np.linalg.norm(xy[i] - xy[j] + b),
                    np.linalg.norm(xy[i] - xy[j] + a - b),
                    np.linalg.norm(xy[i] - xy[j] + a),
                    np.linalg.norm(xy[i] - xy[j] + a + b),
                )
            for k in range(max_nn):
                if d < borders[k]:
                    neighbors[i][k].append(j)
                    break

    return neighbors


def show_arrangement(
    sim: TriangleSimulator,
    ax: Axes,
    color: str = "red",
    marker: str = "o",
    markersize: int = 10,
    show_neighbors: bool = False,
    max_nn: int = 9,
) -> Axes:
    """粒子配置を描画する

    Args:
        sim (TriangleSimulator): シミュレータ
        ax (Axes): 描画するAxes
        color (str, optional): Defaults to "red".
        marker (str, optional): Defaults to "o".
        markersize (int, optional): Defaults to 10.
        show_neighbors (bool, optional): Trueなら近接粒子を結ぶ線を描く. Defaults to False.
        max_nn (int, optional): 第何近接まで線を描くか. Defaults to 9.

    Returns:
        Axes: 描画されたAxes
    """
    ax = _draw_lattice(ax, sim.Lx, sim.Ly, sim.a)

    x, y = sim.real_coords.T
    ax.scatter(x, y, color=color, marker=marker, s=markersize)
    on_bottom = y < sim.a * 0.5
    ax.scatter(
        x[on_bottom] - sim.Ly * sim.a * 0.5,
        y[on_bottom] + sim.Ly * sim.a * _SQRT3BY2,
        color=color,
        marker=marker,
        s=markersize,
    )
    on_left = x < -y * _SQRT3BY2 + 0.5 * sim.a
    ax.scatter(
        x[on_left] + sim.Lx * sim.a,
        y[on_left],
        color=color,
        marker=marker,
        s=markersize,
    )

    if not show_neighbors:
        return ax

    neighbor_colors = (
        "black",
        "purple",
        "brown",
        "red",
        "blue",
        "green",
        "orange",
        "magenta",
        "cyan",
        "pink",
    )

    neighbors = _detect_neighbors(sim, max_nn)
    need_legend = [False for k in range(max_nn)]
    for i in range(sim.n):
        for k in range(max_nn):
            color = neighbor_colors[k]
            for j in neighbors[i][k]:
                if i <= j:
                    pass
                ax.plot([x[i], x[j]], [y[i], y[j]], color=color)
                need_legend[k] = True

    for k in range(max_nn):
        if need_legend[k]:
            ax.scatter([], [], color=neighbor_colors[k], label=f"{k + 1}NN")
    ax.legend()

    return ax


def show_iq_history(sim: TriangleSimulator, ax: Axes, n: int = 10) -> Axes:
    """ステップごとのI(q)を描画する

    Args:
        sim (TriangleSimulator): シミュレータ
        ax (Axes): 描画するAxes
        n (int, optional): 何回分描画するか(総ステップ数に合わせて等間隔にn回分描く). Defaults to 10.

    Returns:
        Axes: 描画したAxes
    """
    ax.plot(sim.q, sim.i_exp / sim.i_exp.sum(), label="exp.")
    steps = range(0, sim.logged_steps.size)
    if n < sim.logged_steps.size:
        steps = np.linspace(0, sim.logged_steps.size - 1, n).astype(int)
    for i in steps:
        step = sim.logged_steps[i]
        x, y = sim.x_log[i], sim.y_log[i]
        iq = sim.compute_i(x, y)
        ax.plot(sim.q, iq / iq.sum(), label=f"step {step}")
    ax.set_xlabel(r"$q$ [nm$^{-1}$]")
    ax.set_ylabel(r"$I(q) [a.u.]$")
    ax.legend()
    return ax
