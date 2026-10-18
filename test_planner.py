"""Checks of the attngraph step on the shipped checkpoints."""

import math

import numpy as np
import planner


def _features(peds: np.ndarray) -> dict:
    return {
        "robot_pose": np.array([0.0, 0.0, 0.0]),
        "robot_state": np.zeros(6),
        "goal_pose": np.array([4.0, 0.0, 0.0]),
        "pedestrians": peds,
    }


def test_full_crowd_out_of_sensor_range_steps() -> None:
    peds = np.array([[i, 10.0 + i, 10.0, 0.0, 0.0] for i in range(planner._MAX_HUMAN_NUM)], dtype=np.float32)
    planner.on_reset("e1", None)
    for _ in range(planner._OBS_SEQ_LEN + 1):
        action = planner.step(_features(peds))
    assert len(action) == 2
    assert all(math.isfinite(v) for v in action)


def test_nearest_pedestrians_take_the_slots_in_a_crowd_over_capacity() -> None:
    far = [[i, 20.0 + i, 20.0, 0.0, 0.0] for i in range(planner._MAX_HUMAN_NUM)]
    peds = np.array([*far, [99, 1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    planner.on_reset("e1", None)
    planner.step(_features(peds))
    assert 99 in planner._ped_id_slot
    assert len(planner._ped_id_slot) == planner._MAX_HUMAN_NUM
