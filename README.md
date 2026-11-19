# 空间音频：声像、多普勒、遮挡与设备

空间音频在声像、距离衰减、多普勒、遮挡和设备配置上出错，导致方向感错误或远近失真。本仓库固定空间化契约。

- 领域：游戏、图形与多媒体
- 子方向：音视频处理
- 语言/框架：Python 3, pytest
- 初始状态：15 个参数化用例全部失败

## 目录结构

```
README.md
conftest.py
.gitignore
audio/__init__.py
audio/spatial_audio.py
tests/test_spatial_audio.py
```

## 需要修复的 10 个问题

1. 声像按方位角等功率分配到左右声道
2. 距离衰减按最小/最大距离钳制
3. 多普勒按听者与声源速度计算
4. 遮挡按墙体数量与吸声衰减
5. 耳机和扬声器使用不同空间化配置
6. 听者变换变化后重新计算声像
7. 多声源各自保留独立空间参数
8. 采样率变化后多普勒因子不漂移
9. 异步声源位置更新做时间对齐
10. 静音声源不参与声像计算

## 15 条验收契约

    01. gains center — `spatial_gains`
    02. gains left — `spatial_gains`
    03. gains right — `spatial_gains`
    04. gains near — `spatial_gains`
    05. gains far — `spatial_gains`
    06. doppler static — `doppler`
    07. doppler approach — `doppler`
    08. doppler recede — `doppler`
    09. doppler listener — `doppler`
    10. doppler half — `doppler`
    11. occlusion one — `occlusion_gain`
    12. occlusion two — `occlusion_gain`
    13. occlusion zero — `occlusion_gain`
    14. occlusion speakers — `occlusion_gain`
    15. occlusion many — `occlusion_gain`

## 当前状态

`audio/spatial_audio.py` 只保留函数签名与文档字符串，所有函数体统一
`raise NotImplementedError`。`tests/test_spatial_audio.py` 使用一个
`@pytest.mark.parametrize` 装载 15 条契约，初始全部失败且没有 collection error。

```bash
python -m pytest -q
```

预期输出包含 `15 failed`。实现 `audio/spatial_audio.py` 后，15 条契约应全部通过。
