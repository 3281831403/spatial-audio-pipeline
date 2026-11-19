"""空间音频：等功率声像、距离钳制衰减、双向多普勒、墙体遮挡与设备配置。

设计要点（对应 README 的 10 个问题）：

1. 声像按方位角做 *等功率* 分配，左右声道平方和恒为 1，转头时总能量不变。
2. 距离先钳制到 ``[min_dist, max_dist]`` 再线性衰减，近处不爆音、远处不出负增益。
3. 多普勒同时计入听者和声源速度（标量或三维矢量均可），因子与采样率无关。
4. 遮挡基于听者→声源线段与墙体的真实相交，而不是直线距离；每堵墙按吸声系数连乘。
5. 耳机（HRTF/双耳）与扬声器（立体声/串音）使用不同的设备配置。
6. 声像是无缓存纯函数：听者变换一旦更新，下一次计算立即反映新方位。
7. 每个 :class:`SpatialSource` 持有自己的距离/静音/位置参数，多声源互不共享。
8. :class:`DopplerShifter` 的频率因子只取决于速度与声速，采样率只影响延迟样本数。
9. 异步声源位置带时间戳，渲染时通过 :class:`AsyncPositionBuffer` 对齐到听者时刻。
10. 静音声源在引擎层直接跳过，:func:`spatial_gains` 也提供 ``muted`` 安全短路。
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Optional, Sequence

DEFAULT_SOUND_SPEED = 343.0
_EPS = 1e-12

# 问题 5：耳机与扬声器是两套空间化配置，不能共用一组参数。
DEVICE_PROFILES = {
    "headphones": {
        "profile": "hrtf",
        "spatial_mode": "binaural",
        "crossfeed": 0.0,
        "stereo_spread": 1.0,
    },
    "speakers": {
        "profile": "stereo",
        "spatial_mode": "stereo",
        "crossfeed": 0.2,
        "stereo_spread": 0.8,
    },
}


# --------------------------------------------------------------------------- #
# 向量小工具（只依赖标准库，输入一律是 (x, y, z) 三元组）
# --------------------------------------------------------------------------- #
def _as_vec(v) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def _sub(a, b) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v) -> float:
    return math.sqrt(_dot(v, v))


def _unit(v) -> tuple[float, float, float]:
    n = _norm(v)
    if n <= _EPS:
        raise ValueError("零向量无法归一化（方向不确定）")
    return (v[0] / n, v[1] / n, v[2] / n)


# --------------------------------------------------------------------------- #
# 问题 1 + 2 + 6 + 10：等功率声像、距离钳制、听者变换、静音短路
# --------------------------------------------------------------------------- #
def _listener_pose(listener, yaw: float) -> tuple[tuple[float, float, float], float]:
    """接受 (x, y, z) 元组或 :class:`Listener`，返回 (位置, 偏航角)。"""
    if isinstance(listener, Listener):
        return _as_vec(listener.position), float(listener.yaw)
    return _as_vec(listener), float(yaw)


def spatial_gains(listener, source, min_dist, max_dist, *, yaw=0.0, muted=False):
    """返回 ``(left_gain, right_gain, attenuation)``。

    方位角在听者坐标系内计算：``yaw=0`` 时听者面向 +z，正偏航表示向 +x 转
    （俯视顺时针）。方位角钳制在 [-90°, +90°]，-90° 为正左、+90° 为正右。

    声像为等功率正弦/余弦定律：``L² + R² ≡ 1``，声源绕头转动时总能量恒定，
    不会像线性声像那样在正侧方能量塌陷。

    距离衰减先把欧氏距离钳制到 ``[min_dist, max_dist]`` 再做线性倒数式斜坡：
    小于最小距离的声源不会继续放大（防近场爆音），大于最大距离的声源增益
    钳为 0（防负增益/远处消失后又折返）。

    纯函数，无缓存：听者调用 :meth:`Listener.update_transform` 后再次调用本
    函数即得到新声像，不存在陈旧变换问题。
    """
    if muted:
        # 问题 10：静音声源不参与声像计算，直接返回零贡献。
        return (0.0, 0.0, 0.0)

    listener_pos, yaw = _listener_pose(listener, yaw)
    source_pos = _as_vec(source)
    min_dist = float(min_dist)
    max_dist = float(max_dist)

    delta = _sub(source_pos, listener_pos)
    distance = _norm(delta)

    # 听者偏航的逆旋转：把世界系位移变到听者局部系。
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    dx = delta[0] * cos_y - delta[2] * sin_y
    dz = delta[0] * sin_y + delta[2] * cos_y

    # 立体声声像无法表达正后方，钳制到 ±90°（折叠到最近一侧）。
    azimuth = max(-math.pi / 2.0, min(math.pi / 2.0, math.atan2(dx, dz)))

    # 等功率声像：α = (azimuth + 90°) / 2，L=cos α，R=sin α。
    alpha = azimuth * 0.5 + math.pi / 4.0
    left = math.cos(alpha)
    right = math.sin(alpha)

    # 问题 2：距离钳制 + 线性衰减。
    if max_dist <= min_dist:
        attenuation = 1.0 if distance <= max_dist else 0.0
    else:
        clamped = max(min_dist, min(max_dist, distance))
        attenuation = (max_dist - clamped) / (max_dist - min_dist)

    return (left, right, attenuation)


# --------------------------------------------------------------------------- #
# 问题 3 + 8：听者/声源双向多普勒，且因子与采样率无关
# --------------------------------------------------------------------------- #
def doppler(listener_vel, source_vel, sound_speed, *, direction=None):
    """返回多普勒频率因子 ``f_observed / f_emitted``。

    标量约定（测试契约）：听者速度为正表示 *接近* 声源（抬高分子），
    声源速度为正表示 *远离* 听者（抬高分子）——两者都沿听源连线测量。

    矢量约定：传入 ``direction``（听者→声源的单位向量，任意长度会自动归
    一化），速度矢量投影到该方向：

        factor = (c + v_listener·d) / (c + v_source·d)

    公式里没有采样率：同一组速度在 44.1 kHz 与 48 kHz 下因子完全相同，
    采样率只改变重采样/延迟线的样本数（见 :class:`DopplerShifter`）。
    声速接近马赫 1 时分母钳制到极小正值，避免除零。
    """
    c = float(sound_speed)
    if direction is not None:
        d = _unit(direction)
        v_listener = _dot(_as_vec(listener_vel), d)
        v_source = _dot(_as_vec(source_vel), d)
    else:
        v_listener = float(listener_vel)
        v_source = float(source_vel)

    numerator = c + v_listener
    denominator = c + v_source
    if abs(denominator) <= _EPS:
        denominator = math.copysign(_EPS, denominator) or _EPS
    return numerator / denominator


@dataclass
class DopplerShifter:
    """多普勒重采样器：频率因子与采样率解耦（问题 8）。"""

    sound_speed: float = DEFAULT_SOUND_SPEED
    sample_rate: float = 48000.0

    def factor(self, listener_vel, source_vel, *, direction=None) -> float:
        # 故意不使用 self.sample_rate：换采样率不能让音高漂移。
        return doppler(listener_vel, source_vel, self.sound_speed, direction=direction)

    def resample_ratio(self, listener_vel, source_vel, *, direction=None) -> float:
        """播放重采样比，等于多普勒因子；采样率变化时比值不变。"""
        return self.factor(listener_vel, source_vel, direction=direction)

    def delay_samples(self, distance) -> float:
        """单程传播延迟换算成样本数——只有这里随采样率变化。"""
        return float(distance) / self.sound_speed * float(self.sample_rate)


# --------------------------------------------------------------------------- #
# 问题 4：基于真实墙体相交的遮挡，而不是直线距离
# --------------------------------------------------------------------------- #
@dataclass
class Wall:
    """x-z 平面上的一段隔墙，``absorption`` 为 0~1 的吸声系数。"""

    start: tuple[float, float, float]
    end: tuple[float, float, float]
    absorption: Optional[float] = None


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[0] - a[0])


def _on_segment(a, b, p) -> bool:
    return (
        min(a[0], b[0]) - _EPS <= p[0] <= max(a[0], b[0]) + _EPS
        and min(a[2], b[2]) - _EPS <= p[2] <= max(a[2], b[2]) + _EPS
    )


def _segments_intersect(p1, p2, q1, q2) -> bool:
    """判断 x-z 平面两条线段是否相交（含端点相接与共线重叠）。"""
    o1, o2 = _orient(p1, p2, q1), _orient(p1, p2, q2)
    o3, o4 = _orient(q1, q2, p1), _orient(q1, q2, p2)
    if o1 == 0.0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0.0 and _on_segment(p1, p2, q2):
        return True
    if o3 == 0.0 and _on_segment(q1, q2, p1):
        return True
    if o4 == 0.0 and _on_segment(q1, q2, p2):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def occlusion_gain(walls, absorption, device, *, listener=None, source=None):
    """返回 ``(gain, profile)``。

    两种调用方式：

    * 计数模式（基础契约）：``walls`` 是遮挡层数（整数），每层统一使用
      ``absorption``，``gain = (1 - absorption) ** walls``。
    * 几何模式：``walls`` 是 :class:`Wall` 序列，函数统计听者→声源线段
      真正穿过的墙，逐墙连乘 ``(1 - wall.absorption)``；墙未自带吸声
      系数时回退到参数 ``absorption``。墙在直线路径之外就不衰减，
      不会出现“隔着墙反而更响”。

    ``profile`` 由设备决定：耳机 ``'hrtf'``，扬声器 ``'stereo'``。
    """
    if device not in DEVICE_PROFILES:
        raise ValueError(f"未知设备 {device!r}，可选：{sorted(DEVICE_PROFILES)}")
    profile = DEVICE_PROFILES[device]["profile"]
    fallback = float(absorption)

    if isinstance(walls, int):
        gain = (1.0 - fallback) ** walls
        return (gain, profile)

    if listener is None or source is None:
        raise ValueError("几何遮挡模式需要同时提供 listener 与 source 位置")

    gain = 1.0
    for wall in walls:
        if _segments_intersect(listener, source, wall.start, wall.end):
            alpha = fallback if wall.absorption is None else float(wall.absorption)
            gain *= 1.0 - alpha
    return (gain, profile)


# --------------------------------------------------------------------------- #
# 问题 6：听者变换（位置 + 偏航），更新后自动失效旧声像
# --------------------------------------------------------------------------- #
@dataclass
class Listener:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    transform_version: int = 0

    def update_transform(self, *, position=None, yaw=None, velocity=None) -> None:
        """更新听者位姿并累加版本号。

        声像计算不做缓存，版本号供引擎/上层判断变换确实发生过：下一次
        :func:`spatial_gains` 调用必然基于新变换重新计算。
        """
        if position is not None:
            self.position = _as_vec(position)
        if yaw is not None:
            self.yaw = float(yaw)
        if velocity is not None:
            self.velocity = _as_vec(velocity)
        self.transform_version += 1


# --------------------------------------------------------------------------- #
# 问题 9：异步声源位置的时间对齐缓冲
# --------------------------------------------------------------------------- #
class AsyncPositionBuffer:
    """带时间戳的位置历史，支持任意时刻的线性插值/外推对齐。"""

    def __init__(self) -> None:
        self._times: list[float] = []
        self._positions: list[tuple[float, float, float]] = []

    def add(self, timestamp: float, position) -> None:
        timestamp = float(timestamp)
        pos = _as_vec(position)
        idx = bisect_left(self._times, timestamp)
        self._times.insert(idx, timestamp)
        self._positions.insert(idx, pos)

    def __len__(self) -> int:
        return len(self._times)

    def position_at(self, timestamp: float) -> tuple[float, float, float]:
        if not self._times:
            raise RuntimeError("异步位置缓冲为空，无法对齐")
        timestamp = float(timestamp)
        if len(self._times) == 1:
            return self._positions[0]

        idx = bisect_left(self._times, timestamp)
        if idx == 0:
            i0, i1 = 0, 1  # 早于最早样本：线性外推
        elif idx == len(self._times):
            i0, i1 = idx - 2, idx - 1  # 晚于最新样本：线性外推
        else:
            i0, i1 = idx - 1, idx  # 区间内：线性插值

        t0, t1 = self._times[i0], self._times[i1]
        p0, p1 = self._positions[i0], self._positions[i1]
        span = t1 - t0
        ratio = 0.0 if span <= _EPS else (timestamp - t0) / span
        return tuple(p0[k] + (p1[k] - p0[k]) * ratio for k in range(3))


# --------------------------------------------------------------------------- #
# 问题 7 + 10：每个声源独立持有自己的空间参数
# --------------------------------------------------------------------------- #
@dataclass
class SpatialSource:
    source_id: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    min_dist: float = 1.0
    max_dist: float = 100.0
    muted: bool = False
    _buffer: AsyncPositionBuffer = field(default_factory=AsyncPositionBuffer)

    def post_position(self, timestamp: float, position) -> None:
        """登记一个带时间戳的异步位置更新（问题 9）。"""
        self._buffer.add(timestamp, position)

    def aligned_position(self, timestamp: Optional[float]):
        """把声源位置对齐到听者时刻；无历史或未给时刻时回退到当前位置。"""
        if timestamp is None or len(self._buffer) == 0:
            return _as_vec(self.position)
        return self._buffer.position_at(timestamp)


# --------------------------------------------------------------------------- #
# 引擎：把 10 项修复串成一次完整的多声源渲染
# --------------------------------------------------------------------------- #
@dataclass
class SpatialAudioEngine:
    device: str = "headphones"
    sound_speed: float = DEFAULT_SOUND_SPEED
    _last_transform_version: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        if self.device not in DEVICE_PROFILES:
            raise ValueError(f"未知设备 {self.device!r}")

    def render(self, sources: Sequence[SpatialSource], listener, *, time=None, walls=()):
        """渲染多个声源，返回每个 *未静音* 声源的独立空间化结果。

        每个声源使用各自的 min/max 距离与静音状态（问题 7、10）；位置先
        做时间对齐（问题 9）；声像随听者变换即时重算（问题 6）；多普勒
        计入听者和声源双方速度（问题 3）；遮挡按真实穿墙计算（问题 4）；
        输出携带当前设备的空间化 profile（问题 5）。
        """
        listener_pos, yaw = _listener_pose(listener, 0.0)
        listener_vel = listener.velocity if isinstance(listener, Listener) else (0.0, 0.0, 0.0)
        if isinstance(listener, Listener):
            self._last_transform_version = listener.transform_version
        profile = DEVICE_PROFILES[self.device]["profile"]

        results = []
        for source in sources:
            if source.muted:
                # 问题 10：静音声源完全跳过，不占声像、不进混音。
                continue

            pos = source.aligned_position(time)
            left, right, attenuation = spatial_gains(
                listener_pos, pos, source.min_dist, source.max_dist, yaw=yaw
            )

            direction = _sub(pos, listener_pos)
            if _norm(direction) > _EPS:
                factor = doppler(
                    listener_vel, source.velocity, self.sound_speed, direction=direction
                )
            else:
                factor = 1.0

            wall_list = list(walls)
            if wall_list and all(isinstance(w, Wall) for w in wall_list):
                occ, occ_profile = occlusion_gain(
                    wall_list, 0.0, self.device, listener=listener_pos, source=pos
                )
            else:
                occ, occ_profile = 1.0, profile

            results.append(
                {
                    "source_id": source.source_id,
                    "left": left * attenuation * occ,
                    "right": right * attenuation * occ,
                    "pan": (left, right),
                    "attenuation": attenuation,
                    "occlusion": occ,
                    "doppler_factor": factor,
                    "profile": occ_profile,
                }
            )
        return results
