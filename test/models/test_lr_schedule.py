from types import SimpleNamespace

import pytest
import torch

from point_frustums.models.point_frustums import PointFrustums


def runtime(lr: float, warmup_steps: int, schedule: str = "constant") -> SimpleNamespace:
    """The attributes that the schedule methods of PointFrustums use, without building the model."""
    module = SimpleNamespace(lr=lr, lr_schedule=schedule, warmup_steps=warmup_steps)
    module._warmup_factor = lambda step: PointFrustums._warmup_factor(module, step)
    return module


@pytest.mark.parametrize("warmup_steps", [0, 1, 4])
def test_constant_schedule_warms_up_linearly(warmup_steps: int):
    module = runtime(lr=2e-3, warmup_steps=warmup_steps)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=module.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=module._warmup_factor)
    lrs = []
    for _ in range(6):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    expected = [2e-3 * min(1.0, (s + 1) / warmup_steps) if warmup_steps else 2e-3 for s in range(6)]
    assert lrs == pytest.approx(expected)


def test_resumption_applies_the_configured_learning_rate():
    """A resumed run restores the scheduler of the checkpoint (here: lr 2e-3 after 10 steps); on_train_start replaces it
    with the configured lr, also in the middle of the warmup."""
    old = runtime(lr=2e-3, warmup_steps=20)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=old.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=old._warmup_factor)
    for _ in range(10):
        optimizer.step()
        scheduler.step()

    new = runtime(lr=5e-4, warmup_steps=20)
    scheduler.lr_lambdas = [new._warmup_factor]
    new.trainer = SimpleNamespace(lr_scheduler_configs=[SimpleNamespace(scheduler=scheduler)])
    PointFrustums.on_train_start(new)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-4 * 11 / 20)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-4 * 12 / 20)


def test_one_cycle_is_left_untouched_on_resumption():
    module = runtime(lr=5e-4, warmup_steps=0, schedule="one_cycle")
    module.trainer = SimpleNamespace(lr_scheduler_configs=[SimpleNamespace(scheduler=None)])
    PointFrustums.on_train_start(module)  # must not access the scheduler
