# OT target assignment ablation

Short training runs that compare settings of the optimal transport target assignment. Each file is the complete
configuration that LightningCLI resolved for the run (`--print_config`); no other config file is needed.

All runs share the following setup:

- 4 epochs, OneCycle schedule with a peak learning rate of 5e-3, and validation after epochs 2 and 4.
- NuScenes trainval with the sensor data of parts 1-3, read as a streaming dataset (`build_streaming --parts 1-3`).
- Augmentations for training only: point subsampling (drop ratio up to 0.4), horizontal flip (p 0.5) and rotation
  (p 0.8). Validation only normalizes the inputs.
- `seed_everything: 42`, and the validation split is the fixed NuScenes val split, read in order (no shuffling).
- Code at commit 582357e.

| Run | Target assignment, differences to the baseline |
|---|---|
| `ot-baseline` | none: epsilon 0.1, rho (1, 1), min_k 5, max_k 30, alpha 1.5, beta 1, gamma 1.2, kappa 4 |
| `ot-eps0.3` | epsilon 0.3 |
| `ot-mink1` | min_k 1 |
| `ot-rho10` | rho (10, 10) |
| `ot-beta2-kappa2` | beta 2, kappa 2 |

Run one of them from the repository root:

```sh
python -m point_frustums fit --config configs/experiments/ot-ablation/ot-baseline.yaml
```

Logs and checkpoints are written to `training_logs/<run>`.

The runs were chosen after a screen of the assignment statistics on a fixed checkpoint. Most settings trade the number
of positives against their quality. rho, epsilon and min_k have the strongest effect. Changes to rho_1, max_k (at
rho 1), the receptive field bounds and the stopping threshold barely change the assignment.
