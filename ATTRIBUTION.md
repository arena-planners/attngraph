Policy adapted from Shuijing725/CrowdNav_Prediction_AttnGraph
(commit 390773137be04ed14e27620dc5fd7c5e1a5b1f62).
Source: https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph
Paper: "Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph",
Liu et al., ICRA 2023, https://arxiv.org/abs/2203.01821
License: MIT (see LICENSE).

## Fork patches

- `arguments.py`: `get_args()` now accepts an optional `argv` list. Upstream called
  `parser.parse_args()` at the bottom, which crashes inside the bridge subprocess because
  the subprocess inherits the host's argv. Library callers pass `get_args([])`.
- `rl/networks/network_utils.py`: removed the top-level `from rl.networks.envs import
  VecNormalize` import. Upstream `rl/networks/envs.py` pulls in `baselines`,
  `dm_control2gym`, and `rl/vec_env/vec_pretext_normalize.py` (the latter chains into
  `gst_updated/scripts/`); none are needed for inference. The import is now lazy inside
  `get_vec_normalize`, which the inference path never calls.
- Stripped training-only modules: `rl/ppo/`, `rl/vec_env/`, `rl/networks/envs.py`,
  `rl/networks/dummy_vec_env.py`, `rl/networks/shmem_vec_env.py`, `rl/networks/storage.py`,
  `rl/evaluation.py`, the `crowd_sim/` env classes, all `crowd_nav/policy/*` except
  `policy.py`, and `gst_updated/src/{mgnn,pec_net}` plus `gst_updated/scripts/`. The
  remaining tree contains only the modules the inference path actually imports.
