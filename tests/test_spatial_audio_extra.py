"""覆盖原 15 条参数契约之外的 7 项架构修复。

对应 README 问题：5（设备区分）、6（听者变换重算）、7（多源独立）、
8（采样率无关）、9（异步时间对齐）、10（静音跳过），并加强 1/3/4 的
不变量（等功率能量、双向多普勒、几何穿墙）。
"""

import math

import pytest

from audio import spatial_audio as mod
from audio.spatial_audio import (
    AsyncPositionBuffer,
    DEVICE_PROFILES,
    DopplerShifter,
    Listener,
    SpatialAudioEngine,
    SpatialSource,
    Wall,
)


# --------------------------------------------------------------------------- #
# 问题 1：等功率声像——任意方位 L²+R² 恒为 1，线性声像做不到
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("deg", [-90, -60, -30, 0, 30, 60, 90])
def test_equal_power_energy_invariant(deg):
    theta = math.radians(deg)
    src = (5 * math.sin(theta), 0.0, 5 * math.cos(theta))
    left, right, _ = mod.spatial_gains((0, 0, 0), src, 1.0, 10.0)
    assert left**2 + right**2 == pytest.approx(1.0)


def test_pan_endpoints_and_center():
    full_left = mod.spatial_gains((0, 0, 0), (-5, 0, 0), 1.0, 10.0)
    full_right = mod.spatial_gains((0, 0, 0), (5, 0, 0), 1.0, 10.0)
    assert full_left[:2] == pytest.approx((1.0, 0.0))
    assert full_right[:2] == pytest.approx((0.0, 1.0))
    center = mod.spatial_gains((0, 0, 0), (0, 0, 5), 1.0, 10.0)
    assert center[:2] == pytest.approx((math.sqrt(0.5), math.sqrt(0.5)))


# --------------------------------------------------------------------------- #
# 问题 2：距离钳制的单调性与边界
# --------------------------------------------------------------------------- #
def test_attenuation_monotonic_and_clamped():
    gains = [
        mod.spatial_gains((0, 0, 0), (0, 0, d), 1.0, 10.0)[2]
        for d in (0.25, 0.5, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0)
    ]
    # 近于 min_dist 一律 1.0（不爆音），远于 max_dist 一律 0.0（不消失成负值）
    assert gains[0] == gains[1] == gains[2] == 1.0
    assert gains[-1] == gains[-2] == 0.0
    for a, b in zip(gains, gains[1:]):
        assert a >= b


# --------------------------------------------------------------------------- #
# 问题 3 + 8：双向多普勒的矢量形式与采样率无关
# --------------------------------------------------------------------------- #
def test_vector_doppler_matches_scalar_contract():
    direction = (0.0, 0.0, 1.0)  # 听者→声源
    approach = mod.doppler((0, 0, 10.0), (0, 0, 0.0), 343.0, direction=direction)
    recede = mod.doppler((0, 0, 0.0), (0, 0, 10.0), 343.0, direction=direction)
    assert approach == pytest.approx(1.0291545189504374)
    assert recede == pytest.approx(0.9716713881019831)


def test_doppler_factor_sample_rate_independent():
    s44 = DopplerShifter(sample_rate=44100.0)
    s48 = DopplerShifter(sample_rate=48000.0)
    f44 = s44.factor(10.0, -5.0)
    f48 = s48.factor(10.0, -5.0)
    assert f44 == f48 == pytest.approx((343.0 + 10.0) / (343.0 - 5.0))
    # 只有延迟样本数随采样率变化
    assert s44.delay_samples(10.0) != s48.delay_samples(10.0)
    assert s48.delay_samples(10.0) == pytest.approx(10.0 / 343.0 * 48000.0)


# --------------------------------------------------------------------------- #
# 问题 4：几何遮挡——只衰减真正穿墙的路径，墙不在路径上不得衰减
# --------------------------------------------------------------------------- #
def test_occlusion_uses_real_wall_intersection():
    listener, source = (0.0, 0.0, 0.0), (10.0, 0.0, 0.0)
    blocking = Wall((5.0, 0.0, -2.0), (5.0, 0.0, 2.0), absorption=0.5)
    off_path = Wall((15.0, 0.0, -2.0), (15.0, 0.0, 2.0), absorption=0.5)

    gain, profile = mod.occlusion_gain(
        [blocking], 0.5, "headphones", listener=listener, source=source
    )
    assert gain == pytest.approx(0.5) and profile == "hrtf"

    gain, _ = mod.occlusion_gain(
        [off_path], 0.5, "headphones", listener=listener, source=source
    )
    assert gain == pytest.approx(1.0)  # 墙在声源后方，直线距离更近也不能误衰减

    gain, _ = mod.occlusion_gain(
        [blocking, Wall((7.0, 0.0, -1.0), (7.0, 0.0, 1.0), absorption=0.25)],
        0.5,
        "headphones",
        listener=listener,
        source=source,
    )
    assert gain == pytest.approx(0.5 * 0.75)


# --------------------------------------------------------------------------- #
# 问题 5：耳机与扬声器使用不同空间化配置
# --------------------------------------------------------------------------- #
def test_device_profiles_are_distinct():
    assert DEVICE_PROFILES["headphones"]["profile"] == "hrtf"
    assert DEVICE_PROFILES["speakers"]["profile"] == "stereo"
    assert DEVICE_PROFILES["headphones"] != DEVICE_PROFILES["speakers"]
    assert mod.occlusion_gain(0, 0.5, "speakers")[1] == "stereo"
    with pytest.raises(ValueError):
        SpatialAudioEngine(device="soundbar")


# --------------------------------------------------------------------------- #
# 问题 6：听者变换更新后声像必须重算（纯函数 + 版本号）
# --------------------------------------------------------------------------- #
def test_listener_transform_changes_pan():
    listener = Listener(position=(0.0, 0.0, 0.0), yaw=0.0)
    before = mod.spatial_gains(listener, (0.0, 0.0, 5.0), 1.0, 10.0)
    listener.update_transform(yaw=math.pi / 2.0)  # 向 +x 转 90°，正前方声源落到左侧
    after = mod.spatial_gains(listener, (0.0, 0.0, 5.0), 1.0, 10.0)
    assert after[:2] == pytest.approx((1.0, 0.0))
    assert before[:2] != pytest.approx(after[:2])
    assert listener.transform_version == 1

    # 与“听者不转、声源在等效左前方”的结果一致
    equivalent = mod.spatial_gains((0, 0, 0), (-5.0, 0.0, 0.0), 1.0, 10.0)
    assert after[:2] == pytest.approx(equivalent[:2])


# --------------------------------------------------------------------------- #
# 问题 7 + 10：多声源独立参数；静音声源被引擎跳过
# --------------------------------------------------------------------------- #
def test_multi_source_independent_params():
    engine = SpatialAudioEngine(device="headphones")
    s1 = SpatialSource("a", position=(0.0, 0.0, 5.0), min_dist=1.0, max_dist=10.0)
    s2 = SpatialSource("b", position=(0.0, 0.0, 5.0), min_dist=1.0, max_dist=20.0)
    s3 = SpatialSource("muted", position=(0.0, 0.0, 5.0), muted=True)
    out = engine.render([s1, s2, s3], Listener())

    assert [row["source_id"] for row in out] == ["a", "b"]  # 静音源不参与
    assert out[0]["attenuation"] == pytest.approx(5.0 / 9.0)
    assert out[1]["attenuation"] == pytest.approx(15.0 / 19.0)
    for row in out:
        assert row["profile"] == "hrtf"
        assert row["left"] ** 2 + row["right"] ** 2 == pytest.approx(
            (row["attenuation"] * row["occlusion"]) ** 2
        )


def test_muted_spatial_gains_short_circuits():
    assert mod.spatial_gains(
        (0, 0, 0), (0, 0, 5), 1.0, 10.0, muted=True
    ) == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# 问题 9：异步位置的时间对齐（插值、外推、引擎接线）
# --------------------------------------------------------------------------- #
def test_async_buffer_interpolation_and_extrapolation():
    buf = AsyncPositionBuffer()
    buf.add(0.0, (0.0, 0.0, 0.0))
    buf.add(1.0, (0.0, 0.0, 10.0))
    assert buf.position_at(0.5) == pytest.approx((0.0, 0.0, 5.0))
    assert buf.position_at(2.0) == pytest.approx((0.0, 0.0, 20.0))  # 线性外推
    assert buf.position_at(-1.0) == pytest.approx((0.0, 0.0, -10.0))


def test_engine_aligns_async_source_to_listener_time():
    engine = SpatialAudioEngine()
    src = SpatialSource("moving", min_dist=1.0, max_dist=10.0)
    src.post_position(0.0, (0.0, 0.0, 0.0))
    src.post_position(1.0, (0.0, 0.0, 10.0))
    aligned = engine.render([src], Listener(), time=0.5)[0]
    expected = mod.spatial_gains((0, 0, 0), (0.0, 0.0, 5.0), 1.0, 10.0)
    assert aligned["pan"] == pytest.approx(expected[:2])
    assert aligned["attenuation"] == pytest.approx(expected[2])


# --------------------------------------------------------------------------- #
# 引擎端到端：听者移动 + 声源移动同时驱动多普勒
# --------------------------------------------------------------------------- #
def test_engine_doppler_uses_listener_and_source_velocity():
    engine = SpatialAudioEngine()
    src = SpatialSource(
        "s", position=(0.0, 0.0, 10.0), velocity=(0.0, 0.0, -10.0)  # 接近听者
    )
    listener = Listener(velocity=(0.0, 0.0, 5.0))  # 听者也接近声源
    out = engine.render([src], listener)[0]
    assert out["doppler_factor"] == pytest.approx((343.0 + 5.0) / (343.0 - 10.0))
