"""空间音频：声像、多普勒、遮挡与设备：待实现的核心模块。"""

from __future__ import annotations


def spatial_gains(listener, source, min_dist, max_dist):
    """返回 (left, right, attenuation)。"""
    raise NotImplementedError

def doppler(listener_vel, source_vel, sound_speed):
    """返回多普勒频率因子。"""
    raise NotImplementedError

def occlusion_gain(walls, absorption, device):
    """返回 (gain, profile)。"""
    raise NotImplementedError
