"""Intention-Aware AttnGraph (selfAttn_merge_SRNN + GST) wrapper for arena_planners bridge.

GST input shape (verified by reading gst_updated/scripts/wrapper/crowd_nav_interface_parallel.py
and the args.pickle bundled with the rand checkpoint): the predictor needs a *history window*
of obs_seq_len=5 absolute pedestrian positions, shape (nenv, num_peds, 5, 2), plus a binary
visibility mask (nenv, num_peds, 5, 1). It returns 5 predicted future positions per ped. We
therefore maintain a 5-frame pedestrian history keyed by Arena's pedestrian id; both that
history and the policy's GRU hidden state are cleared in on_reset.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import pickle
import sys

import numpy as np
import torch
from arena_planners.geometry import lookahead_on_path
from arena_planners.sdk import load_manifest, main_loop

# Vendored upstream code lives flat under this directory; expose it on sys.path so the
# upstream's `from rl.networks...` / `from gst_updated...` absolute imports resolve.
sys.path.insert(0, os.path.dirname(__file__))

_MODEL_DIR = pathlib.Path(__file__).parent / "model"
_POLICY_WEIGHTS = _MODEL_DIR / "attngraph_policy.pt"
_GST_WEIGHTS = _MODEL_DIR / "attngraph_gst.pt"
_GST_ARGS = _MODEL_DIR / "attngraph_gst_args.pickle"

_V_PREF: float = 1.0
_RADIUS: float = 0.3
_LOOKAHEAD: float = 2.0
_GOAL_MAX_DIST: float = 8.0
_MAX_HUMAN_NUM: int = 20
_PREDICT_STEPS: int = 5
_OBS_SEQ_LEN: int = 5
_INVALID_POS: float = -999.0
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_actor_critic: torch.nn.Module | None = None
_gst: object | None = None
_rnn_hxs: dict[str, torch.Tensor] | None = None
_traj_buffer: list[np.ndarray] | None = None  # 5x (max_human_num, 2) absolute positions
_mask_buffer: list[np.ndarray] | None = None  # 5x (max_human_num,) visible bools
_ped_id_slot: dict[int, int] | None = None    # ped id -> row in the buffer


def _build_args() -> argparse.Namespace:
    from arguments import get_args

    args = get_args([])
    # Inference is single-process; trained checkpoint hardcodes max_human_num=20 via the
    # spatial_edges shape passed to selfAttn_merge_SRNN.__init__.
    args.num_processes = 1
    args.num_mini_batch = 1
    args.no_cuda = _DEVICE.type == "cpu"
    args.env_name = "CrowdSimPredRealGST-v0"
    args.sort_humans = True
    return args


def _build_policy() -> torch.nn.Module:
    import gym
    from rl.networks.model import Policy

    args = _build_args()
    spatial_edge_dim = 2 * (_PREDICT_STEPS + 1)
    obs_space = gym.spaces.Dict({
        "robot_node": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7), dtype=np.float32),
        "temporal_edges": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2), dtype=np.float32),
        "spatial_edges": gym.spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(_MAX_HUMAN_NUM, spatial_edge_dim), dtype=np.float32),
        "visible_masks": gym.spaces.Box(low=False, high=True, shape=(_MAX_HUMAN_NUM,), dtype=np.bool_),
        "detected_human_num": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
    })
    action_space = gym.spaces.Box(low=-_V_PREF, high=_V_PREF, shape=(2,), dtype=np.float32)
    actor_critic = Policy(obs_space.spaces, action_space, base="selfAttn_merge_srnn", base_kwargs=args)
    state = torch.load(str(_POLICY_WEIGHTS), map_location=_DEVICE)
    # Upstream save format: list [state_dict, ob_rms]; we want the state_dict only.
    if isinstance(state, list):
        state = state[0]
    actor_critic.load_state_dict(state, strict=True)
    actor_critic.to(_DEVICE)
    actor_critic.eval()
    return actor_critic


def _build_gst() -> object:
    from gst_updated.src.gumbel_social_transformer.st_model import st_model

    with open(_GST_ARGS, "rb") as fh:
        gst_args = pickle.load(fh)
    model = st_model(gst_args, device=_DEVICE).to(_DEVICE)
    ckpt = torch.load(str(_GST_WEIGHTS), map_location=_DEVICE)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt, strict=True)
    model.eval()
    return model


def _init_state() -> None:
    global _rnn_hxs, _traj_buffer, _mask_buffer, _ped_id_slot
    args = _build_args()
    # selfAttn_merge_SRNN.forward expects rnn_hxs values in (nenv, agent_num, hidden_size) shape
    # after the per-step squeeze(0) it applies on the way out (selfAttn_srnn_temp_node.py:444).
    _rnn_hxs = {
        "human_node_rnn": torch.zeros(1, 1, args.human_node_rnn_size, device=_DEVICE),
        "human_human_edge_rnn": torch.zeros(1, 1 + _MAX_HUMAN_NUM, args.human_human_edge_rnn_size, device=_DEVICE),
    }
    _traj_buffer = [np.full((_MAX_HUMAN_NUM, 2), _INVALID_POS, dtype=np.float32) for _ in range(_OBS_SEQ_LEN)]
    _mask_buffer = [np.zeros(_MAX_HUMAN_NUM, dtype=bool) for _ in range(_OBS_SEQ_LEN)]
    _ped_id_slot = {}


def _slot_for(ped_id: int) -> int | None:
    """Reserve a stable row in the rolling buffer for a given pedestrian id."""
    if ped_id in _ped_id_slot:
        return _ped_id_slot[ped_id]
    used = set(_ped_id_slot.values())
    for s in range(_MAX_HUMAN_NUM):
        if s not in used:
            _ped_id_slot[ped_id] = s
            return s
    return None


def _run_gst(traj_hist: np.ndarray, mask_hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward GST. traj_hist: (max_human_num, 5, 2) absolute pos; mask_hist: (max_human_num, 5).

    Returns (mu_pred (max_human_num, 5, 2) absolute pos, valid_mask (max_human_num,)).
    """
    in_traj = torch.from_numpy(traj_hist).to(_DEVICE).float().unsqueeze(0)
    in_mask = torch.from_numpy(mask_hist.astype(np.float32)).to(_DEVICE).unsqueeze(0).unsqueeze(-1)
    invalid_value = _INVALID_POS
    obs_traj = in_traj.permute(0, 1, 3, 2)  # (1, n, 2, 5)
    n_env, num_peds = obs_traj.shape[:2]
    loss_mask_obs = in_mask[:, :, :, 0]
    loss_mask_rel_obs = loss_mask_obs[:, :, :-1] * loss_mask_obs[:, :, -1:]
    loss_mask_rel_obs = torch.cat((loss_mask_obs[:, :, :1], loss_mask_rel_obs), dim=2)
    loss_mask_rel_pred = torch.ones((n_env, num_peds, _PREDICT_STEPS), device=_DEVICE) * loss_mask_rel_obs[:, :, -1:]
    loss_mask_rel = torch.cat((loss_mask_rel_obs, loss_mask_rel_pred), dim=2)
    loss_mask_rel_obs_p = loss_mask_rel_obs.permute(0, 2, 1).reshape(n_env * _OBS_SEQ_LEN, num_peds)
    attn_mask_obs = torch.bmm(loss_mask_rel_obs_p.unsqueeze(2), loss_mask_rel_obs_p.unsqueeze(1))
    attn_mask_obs = attn_mask_obs.reshape(n_env, _OBS_SEQ_LEN, num_peds, num_peds)
    obs_traj_rel = obs_traj[:, :, :, 1:] - obs_traj[:, :, :, :-1]
    obs_traj_rel = torch.cat((torch.zeros(n_env, num_peds, 2, 1, device=_DEVICE), obs_traj_rel), dim=3)
    obs_traj_rel = invalid_value * torch.ones_like(obs_traj_rel) * (1 - loss_mask_rel_obs.unsqueeze(2)) \
        + obs_traj_rel * loss_mask_rel_obs.unsqueeze(2)
    v_obs = obs_traj_rel.permute(0, 3, 1, 2)
    seq_p = obs_traj.permute(0, 3, 1, 2)
    a_obs = seq_p.unsqueeze(3) - seq_p.unsqueeze(2)
    with torch.no_grad():
        gaussian_params, _, _ = _gst(v_obs, a_obs, attn_mask_obs, loss_mask_rel,
                                     tau=0.03, hard=True, sampling=False, device=_DEVICE)
    mu = gaussian_params[0]  # (1, pred_seq_len, num_peds, 2)
    mu = mu.cumsum(1) + obs_traj.permute(0, 3, 1, 2)[:, -1:]
    mu = mu.squeeze(0).permute(1, 0, 2).cpu().numpy()  # (num_peds, pred_seq_len, 2)
    valid = loss_mask_rel_pred[0, :, 0].cpu().numpy().astype(bool)
    return mu, valid


def step(features: dict) -> list[float]:
    global _actor_critic, _gst, _rnn_hxs

    if _actor_critic is None:
        _actor_critic = _build_policy()
    if _gst is None:
        _gst = _build_gst()
    if _rnn_hxs is None:
        _init_state()

    robot_pose = features.get("robot_pose")
    robot_state = features.get("robot_state")
    if robot_pose is None or robot_state is None:
        return [0.0, 0.0]
    px, py, theta = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    vx, vy = float(robot_state[2]), float(robot_state[3])

    global_plan = features.get("global_plan")
    goal_pose = features.get("goal_pose")
    target: tuple[float, float] | None = None
    if global_plan is not None and len(global_plan) > 0:
        target = lookahead_on_path(global_plan, robot_pose, lookahead=_LOOKAHEAD)
    if target is None and goal_pose is not None:
        target = (float(goal_pose[0]), float(goal_pose[1]))
    if target is None:
        return [0.0, 0.0]
    gx, gy = target

    # Slot incoming pedestrians into stable rows so the rolling history is per-id.
    peds = features.get("pedestrians")
    if peds is None:
        peds = []
    cur_pos = np.full((_MAX_HUMAN_NUM, 2), _INVALID_POS, dtype=np.float32)
    cur_mask = np.zeros(_MAX_HUMAN_NUM, dtype=bool)
    seen_slots: set[int] = set()
    for ped in peds[:_MAX_HUMAN_NUM]:
        pid = int(ped[0])
        slot = _slot_for(pid)
        if slot is None:
            continue
        cur_pos[slot] = (float(ped[1]), float(ped[2]))
        cur_mask[slot] = True
        seen_slots.add(slot)
    # Free stale slots so id reuse is possible after a long absence.
    for pid, slot in list(_ped_id_slot.items()):
        if slot not in seen_slots:
            del _ped_id_slot[pid]

    _traj_buffer.append(cur_pos)
    _mask_buffer.append(cur_mask)
    del _traj_buffer[0]
    del _mask_buffer[0]
    traj_hist = np.stack(_traj_buffer, axis=1)  # (max_human_num, 5, 2)
    mask_hist = np.stack(_mask_buffer, axis=1)  # (max_human_num, 5)

    pred_pos, pred_valid = _run_gst(traj_hist, mask_hist)

    # The SRNN net consumes absolute coords with no internal rotate and trained with px, py, gx, gy
    # in a ~+/-8.5 m box, so recentre on the robot and clamp the goal offset to stay in-distribution.
    gdx, gdy = gx - px, gy - py
    gdist = float(np.hypot(gdx, gdy))
    if gdist > _GOAL_MAX_DIST:
        gdx, gdy = gdx * _GOAL_MAX_DIST / gdist, gdy * _GOAL_MAX_DIST / gdist
    robot_node = np.array([[0.0, 0.0, _RADIUS, gdx, gdy, _V_PREF, theta]], dtype=np.float32)
    temporal_edges = np.array([[vx, vy]], dtype=np.float32)

    # spatial_edges layout per human: [rel_curr_x, rel_curr_y, rel_step1_x, rel_step1_y, ...,
    # rel_step5_x, rel_step5_y]; rel = abs - robot_pos. Invalid humans default to far-away.
    spatial = np.full((_MAX_HUMAN_NUM, 2 * (_PREDICT_STEPS + 1)), 15.0, dtype=np.float32)
    visible = np.zeros(_MAX_HUMAN_NUM, dtype=bool)
    for i in range(_MAX_HUMAN_NUM):
        if not cur_mask[i]:
            continue
        spatial[i, 0] = cur_pos[i, 0] - px
        spatial[i, 1] = cur_pos[i, 1] - py
        if pred_valid[i]:
            for k in range(_PREDICT_STEPS):
                spatial[i, 2 + 2 * k] = pred_pos[i, k, 0] - px
                spatial[i, 3 + 2 * k] = pred_pos[i, k, 1] - py
        else:
            for k in range(_PREDICT_STEPS):
                spatial[i, 2 + 2 * k] = spatial[i, 0]
                spatial[i, 3 + 2 * k] = spatial[i, 1]
        visible[i] = True

    # Sort by distance to robot (matches VecPretextNormalize.process_obs_rew tail logic).
    dists = np.linalg.norm(spatial[:, :2], axis=1)
    order = np.argsort(dists)
    spatial = spatial[order]
    visible = visible[order]
    # MultiheadAttention NaNs out when every key is padded; upstream sort_humans=False has a
    # dummy_human_mask guard, sort_humans=True does not. Clamp to >=1 so row 0 (the closest
    # far-placeholder ped at 15 m) participates in attention as a no-op visible slot.
    detected_n = max(int(visible.sum()), 1)

    obs = {
        "robot_node": torch.from_numpy(robot_node).to(_DEVICE).unsqueeze(0),
        "temporal_edges": torch.from_numpy(temporal_edges).to(_DEVICE).unsqueeze(0),
        "spatial_edges": torch.from_numpy(spatial).to(_DEVICE).unsqueeze(0),
        "visible_masks": torch.from_numpy(visible).to(_DEVICE).unsqueeze(0),
        "detected_human_num": torch.tensor([[float(detected_n)]], device=_DEVICE),
    }
    masks = torch.ones(1, 1, device=_DEVICE)

    with torch.no_grad():
        _, action, _, new_hxs = _actor_critic.act(obs, _rnn_hxs, masks, deterministic=True)
    _rnn_hxs = new_hxs

    raw = action.squeeze().cpu().numpy()  # holonomic (vx, vy) in world frame
    speed = float(np.linalg.norm(raw))
    if speed > _V_PREF:
        raw = raw / speed * _V_PREF

    return [float(raw[0]), float(raw[1])]


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    global _rnn_hxs, _traj_buffer, _mask_buffer, _ped_id_slot
    _rnn_hxs = None
    _traj_buffer = None
    _mask_buffer = None
    _ped_id_slot = None


if __name__ == "__main__":
    manifest = load_manifest(pathlib.Path(__file__).parent / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset)
