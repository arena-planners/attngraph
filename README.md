# attngraph

Arena wrapper for **Intention-Aware AttnGraph** (selfAttn_merge_SRNN + GST), the Liu et al.
ICRA 2023 crowd navigator. Adapted from
[Shuijing725/CrowdNav_Prediction_AttnGraph](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph).

Two stacked components:

- **GST predictor** (`gst_updated/`): Gumbel Social Transformer forecasting each visible
  pedestrian's position 5 steps (1.25 s) ahead. Requires a 5-frame observation window.
- **selfAttn_merge_SRNN policy** (`rl/networks/`): recurrent value/actor network with
  per-human-node GRUs and human-human spatial attention. Predicted future positions are
  concatenated into each human's spatial edge feature (12-dim per human).

## Run

```sh
arena launch mobile:=drl mobile.planner:=attngraph
```

Requires a global plan. Defaults to `nav2/navfn`.

## Files

- `planner.py`: bridge wrapper. Maintains a 5-frame pedestrian history and the policy's
  recurrent hidden state across timesteps, clears both on `on_reset`.
- `arguments.py`, `rl/`, `gst_updated/`: vendored upstream code, lightly patched
  (see [ATTRIBUTION.md](ATTRIBUTION.md)).
- `planner.yaml`: observation manifest.
- `weights.yaml`: pulls policy + GST checkpoints from
  [arena-rosnav/attngraph](https://huggingface.co/arena-rosnav/attngraph) on
  `arena feature planners add attngraph`.

## License

MIT (inherited from upstream).
