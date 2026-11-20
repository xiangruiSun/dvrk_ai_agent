"""Recorded episodes (with a small generated recording) and, when they have been downloaded,
the real dVRK files."""
from __future__ import annotations

import math

import numpy as np
import pytest

from vxagent.config import get_settings
from vxagent.data import episodes as E
from vxagent.robot.motion import check_goal
from vxagent.tools.executor import execute_tool
from vxagent.llm.base import ToolCall

HEADER = ["Date"] + [f"{arm}_{s}" for arm in ("PSM1", "PSM2")
                     for s in ("position_x", "position_y", "position_z", "velocity_linear_x",
                               "velocity_linear_y", "velocity_linear_z")]


@pytest.fixture()
def recording(tmp_path, monkeypatch):
    """Two seconds at 50 Hz in the ROSMA layout: PSM1 moves 100 mm along x at 50 mm/s, PSM2 is
    still. The last row is the summary row the real files end with."""
    monkeypatch.setenv("VXAGENT_DATA_DIR", str(tmp_path))
    d = tmp_path / "rosma"
    d.mkdir()
    lines = [",".join(HEADER)]
    for i in range(101):
        t = i * 0.02
        stamp = f"2020-02-06.14:49:{int(t):02d}.{int(round((t % 1) * 1000)):03d}"
        psm1 = [0.05 * t, 0.0, -0.1, 0.05, 0.0, 0.0]
        psm2 = [0.0, 0.02, -0.1, 0.0, 0.0, 0.0]
        lines.append(",".join([stamp] + [f"{x:.6f}" for x in psm1 + psm2]))
    lines.append("Date:2020-02-06.14:49:02," + ",".join(["NaN"] * (len(HEADER) - 1)))
    (d / "X01_Wire_Chaser_I_01.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "scores.csv").write_text("File_name,Time,Errors,Score,\nX01_Wire_Chaser_I_01,2.0,1,17.0,\n", encoding="utf-8")
    return d


def test_episode_statistics_are_computed_from_the_recording(recording):
    eps = E.list_episodes()
    assert [(e.operator, e.task, e.trial) for e in eps] == [("X01", "Wire Chaser I", 1)]
    s = E.compute_stats(eps[0])
    assert s["samples"] == 101 and s["duration_s"] == 2.0 and s["sample_rate_hz"] == 50.0
    assert s["psm1_path_length_mm"] == 100.0 and s["psm1_peak_speed_mm_per_s"] == 50.0
    assert s["psm2_path_length_mm"] == 0.0
    assert E.scores()["X01_Wire_Chaser_I_01"] == {"dataset_time_s": 2.0, "dataset_errors": 1, "dataset_score": 17.0}


def test_window_is_bounded_and_in_millimetres(recording):
    w = E.window(E.list_episodes()[0], "PSM1", "position", 1.0, 5)
    assert w["samples_returned"] == 5 and w["unit"] == "mm" and w["t_s"][0] == 1.0
    assert w["values"][0] == [50.0, 0.0, -100.0]


def test_episode_tools_return_statistics_not_the_recording(recording, ready):
    reg, ctx, _ = ready
    out = execute_tool(reg, ToolCall("e", "episode_summary", {"episode_id": "X01_Wire_Chaser_I_01"}), ctx)
    assert out.ok and out.content["duration_s"] == 2.0 and out.content["dataset_errors"] == 1
    assert len(out.observation()) < 1500
    big = execute_tool(reg, ToolCall("w", "read_signal_window", {"episode_id": "X01_Wire_Chaser_I_01", "arm": "PSM1",
                                                                "signal": "position", "start_s": 0, "count": 400}), ctx)
    assert big.error_kind == "invalid_arguments"      # more than 40 samples cannot even be asked for
    missing = execute_tool(reg, ToolCall("m", "episode_summary", {"episode_id": "X99_Nope_01"}), ctx)
    assert missing.error_kind == "invalid_arguments" and "list_episodes" in missing.hint


# ------------------------------------------------------------------------------ real dVRK files

DVRK = get_settings().dvrk_dir / "share"
needs_dvrk = pytest.mark.skipif(not (DVRK / "kinematic" / "PSM.json").exists(),
                                reason="run scripts/fetch_dvrk_config.py to download the dVRK files")


@needs_dvrk
def test_real_psm_with_large_needle_driver():
    from vxagent.robot.site import load_site
    site = load_site("site_a")
    assert [j.name for j in site.chain.joints] == ["yaw", "pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw"]
    T = site.chain.fk(site.home_q)
    # insertion 0.12 - 0.4318 + 0.4162 along the shaft, plus the 0.0091 wrist link
    assert np.allclose(T[:3, 3], [0.0, 0.0, -0.1135], atol=1e-5)
    assert math.isclose(site.chain.joints[2].qmax, 0.24)
    for name, want in {"hover_needle": set(), "pregrasp": set(), "retract": {"workspace"},
                       "pad_touch": {"clearance"}, "wrist_limit": {"joint_margin"}}.items():
        got = {v.rule for v in check_goal(site.profile, site.scene, site.chain, site.poses[name].T, site.home_q)}
        assert got == want, name
