import re
from pathlib import Path

import numpy as np
import pytest

from blink import image_processing as imp
from blink import time_series

TEST_DATA_DIR = Path(__file__).parent / "test_data"


def test_time_traces():

    traces = time_series.TimeTraces(
        channels={
            imp.Channel("blue", "blue"): np.arange(10),
            imp.Channel("green", "green"): np.arange(20) + 0.5,
        },
        n_traces=10,
    )
    assert traces.n_traces == 10

    channels = traces.get_channels()
    assert isinstance(channels, list)
    for i in channels:
        assert isinstance(i, imp.Channel)
    assert channels == [imp.Channel("blue", "blue"), imp.Channel("green", "green")]

    expected_exception_str = "Input array length (11) does not match n_traces (10)."
    with pytest.raises(ValueError, match=re.escape(expected_exception_str)):
        traces.set_value(
            "intensity",
            channel=imp.Channel("blue", "blue"),
            time=0,
            array=np.arange(11),
        )

    fake_data = np.arange(10)
    for i in range(10):
        traces.set_value(
            "intensity",
            channel=imp.Channel("blue", "blue"),
            time=0 + i,
            array=fake_data + i * 2,
        )
    correct_data = np.arange(0, 20, 2)
    for i in range(10):
        intensity = traces.get_intensity(imp.Channel("blue", "blue"), i)
        np.testing.assert_equal(intensity, correct_data + i)

    traces.set_value(
        "is_colocalized",
        channel=imp.Channel("blue", "blue"),
        time=0 + i,
        array=fake_data + i * 2,
    )

    for i in range(10):
        traces.set_value(
            "intensity",
            channel=imp.Channel("green", "green"),
            time=0.5 + i,
            array=100 + fake_data + i * 2,
        )
    path = TEST_DATA_DIR / "save_path/traces"
    assert isinstance(traces.get_time(imp.Channel("green", "green")), np.ndarray)
    assert isinstance(traces.get_time(imp.Channel("blue", "blue")), np.ndarray)
    traces.to_npz(path)
    new_traces = time_series.TimeTraces.from_npz(path.with_suffix(".npz"))
    assert traces.n_traces == new_traces.n_traces
    for i in range(new_traces.n_traces):
        np.testing.assert_equal(
            new_traces.get_intensity(imp.Channel("blue", "blue"), i),
            traces.get_intensity(imp.Channel("blue", "blue"), i),
        )
        np.testing.assert_equal(
            new_traces.get_intensity(imp.Channel("green", "green"), i),
            traces.get_intensity(imp.Channel("green", "green"), i),
        )
        np.testing.assert_equal(
            new_traces.get_time(imp.Channel("blue", "blue")),
            traces.get_time(imp.Channel("blue", "blue")),
        )
        np.testing.assert_equal(
            new_traces.get_time(imp.Channel("green", "green")),
            traces.get_time(imp.Channel("green", "green")),
        )

    assert traces.has_variable(imp.Channel("blue", "blue"), "is_colocalized")
    assert not traces.has_variable(imp.Channel("blue", "blue"), "is_colocalizeds")
    assert not traces.has_variable(imp.Channel("green", "green"), "is_colocalized")
    assert isinstance(new_traces.get_time(imp.Channel("green", "green")), np.ndarray)
    assert isinstance(new_traces.get_time(imp.Channel("blue", "blue")), np.ndarray)
