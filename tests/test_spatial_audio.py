import pytest

from audio import spatial_audio as mod

CASES = [
    pytest.param('spatial_gains', ((0, 0, 0), (0, 0, 5), 1.0, 10.0), (0.7071067811865476, 0.7071067811865476, 0.5555555555555556), True, id='01_gains_center'),
    pytest.param('spatial_gains', ((0, 0, 0), (-5, 0, 5), 1.0, 10.0), (0.9238795325112867, 0.38268343236508984, 0.325436909792725), True, id='02_gains_left'),
    pytest.param('spatial_gains', ((0, 0, 0), (5, 0, 5), 1.0, 10.0), (0.38268343236508984, 0.9238795325112867, 0.325436909792725), True, id='03_gains_right'),
    pytest.param('spatial_gains', ((0, 0, 0), (0, 0, 0.5), 1.0, 10.0), (0.7071067811865476, 0.7071067811865476, 1.0), True, id='04_gains_near'),
    pytest.param('spatial_gains', ((0, 0, 0), (0, 0, 20), 1.0, 10.0), (0.7071067811865476, 0.7071067811865476, 0.0), True, id='05_gains_far'),
    pytest.param('doppler', (0.0, 0.0, 343.0), 1.0, True, id='06_doppler_static'),
    pytest.param('doppler', (0.0, 10.0, 343.0), 0.9716713881019831, True, id='07_doppler_approach'),
    pytest.param('doppler', (0.0, -10.0, 343.0), 1.03003003003003, True, id='08_doppler_recede'),
    pytest.param('doppler', (10.0, 0.0, 343.0), 1.0291545189504374, True, id='09_doppler_listener'),
    pytest.param('doppler', (0.0, 171.5, 343.0), 0.6666666666666666, True, id='10_doppler_half'),
    pytest.param('occlusion_gain', (1, 0.5, 'headphones'), (0.5, 'hrtf'), True, id='11_occlusion_one'),
    pytest.param('occlusion_gain', (2, 0.5, 'headphones'), (0.25, 'hrtf'), True, id='12_occlusion_two'),
    pytest.param('occlusion_gain', (0, 0.5, 'speakers'), (1.0, 'stereo'), True, id='13_occlusion_zero'),
    pytest.param('occlusion_gain', (1, 0.2, 'speakers'), (0.8, 'stereo'), True, id='14_occlusion_speakers'),
    pytest.param('occlusion_gain', (3, 0.1, 'headphones'), (0.7290000000000001, 'hrtf'), True, id='15_occlusion_many'),
]


def assert_close(actual, expected):
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_close(a, e)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_close(a, e)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_close(actual[key], expected[key])
    else:
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize("func_name,args,expected,approx", CASES)
def test_spatial_audio_contract(func_name, args, expected, approx):
    func = getattr(mod, func_name)
    result = func(*args)
    if approx:
        assert_close(result, expected)
    else:
        assert result == expected
