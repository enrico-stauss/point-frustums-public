import warnings

import pytest
import torch
from ot.unbalanced import sinkhorn_unbalanced, sinkhorn_unbalanced_translation_invariant

from point_frustums.ops.sinkhorn import _sinkhorn_torch, sinkhorn_unbalanced_ti, sinkhorn_unbalanced_ti_batch

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")


def assignment_problem(n: int, m: int, seed: int, device: str = "cpu") -> tuple[torch.Tensor, ...]:
    """
    A problem shaped like the target assignment: n predictions with demand 1, m targets with a supply of a few
    predictions each and the background column with the rest. Every prediction has a low cost for a few targets (the
    pre-mapping) and a high cost for all others; the background is a little more expensive than any foreground.
    """
    g = torch.Generator().manual_seed(seed)
    a = torch.ones(n)
    # At most half of the predictions are supplied by the targets, the rest by the background
    supply = (torch.rand(m, generator=g) * 0.8 + 0.2) * min(20.0, 0.5 * n / m)
    b = torch.cat([supply, (n - supply.sum()).reshape(1)])
    pre_mapped = torch.rand(n, m, generator=g) < 0.02
    cost = torch.rand(n, m, generator=g) * 2.0 + 2.2 * (~pre_mapped)
    background = torch.rand(n, 1, generator=g) * 0.5 + 2.5
    M = torch.cat([cost, background], dim=1)
    return a.to(device), b.to(device), M.to(device)


def pot(a, b, M, reg=0.1, stop_thr=1e-3, max_iter=1000):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan, log = sinkhorn_unbalanced_translation_invariant(
            a, b, M, reg, reg_m=(1.0, 1.0), reg_type="kl", stopThr=stop_thr, numItermax=max_iter, log=True
        )
    return plan, len(log["err"])


def assign(plan: torch.Tensor) -> torch.Tensor:
    """As the target assignment: rescale each column to a maximum of 1, the best column of each row."""
    return (plan / plan.max(dim=0).values[None, :]).argmax(dim=1)


def assert_plans_close(plan, reference):
    plan, reference = plan.cpu(), reference.cpu()
    scale = reference.abs().max()
    torch.testing.assert_close(plan / scale, reference / scale, rtol=0, atol=1e-4)


@pytest.mark.parametrize("n, m", [(500, 7), (3000, 40), (1000, 1)])
def test_torch_matches_pot(n: int, m: int):
    a, b, M = assignment_problem(n, m, seed=n + m)
    reference, n_iter_pot = pot(a, b, M)
    plan, n_iter = _sinkhorn_torch(a, b, M, 0.1, (1.0, 1.0), 1e-3, 1000)
    assert int(n_iter) == n_iter_pot
    assert_plans_close(plan, reference)
    assert torch.equal(assign(plan), assign(reference))


def test_converges_to_the_optimum():
    """With a tight threshold, the plan agrees with POT (float64) to high precision."""
    a, b, M = (x.double() for x in assignment_problem(400, 9, seed=1))
    reference, _ = pot(a, b, M, stop_thr=1e-12, max_iter=20000)
    plan, _ = _sinkhorn_torch(a, b, M, 0.1, (1.0, 1.0), 1e-12, 20000)
    torch.testing.assert_close(plan, reference, rtol=1e-8, atol=1e-10)


def optimum(a, b, M, reg_m, reg=0.1):
    """The optimal plan from the classical (not translation invariant) unbalanced Sinkhorn, run to convergence."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sinkhorn_unbalanced(
            a, b, M, reg, reg_m, method="sinkhorn", reg_type="kl", numItermax=200000, stopThr=1e-13
        )


@pytest.mark.parametrize("reg_m", [(1.0, 3.0), (3.0, 1.0), (0.5, 0.5)])
def test_unequal_penalties_converge_to_the_optimum(reg_m):
    """For rho_1 != rho_2, POT's translation invariant iteration does not converge to the optimum (ReadMe.md), the
    exact maximization of the translation invariant dual does."""
    a, b, M = (x.double() for x in assignment_problem(600, 8, seed=2))
    reference = optimum(a, b, M, reg_m)
    plan, _ = _sinkhorn_torch(a, b, M, 0.1, reg_m, 1e-12, 20000)
    torch.testing.assert_close(plan, reference, rtol=1e-7, atol=1e-9)


@mps
@pytest.mark.parametrize("reg_m", [(1.0, 3.0), (3.0, 1.0)])
def test_metal_unequal_penalties_converge_to_the_optimum(reg_m):
    a, b, M = assignment_problem(600, 8, seed=2)
    reference = optimum(a.double(), b.double(), M.double(), reg_m).float()
    plans, _ = sinkhorn_unbalanced_ti_batch([(a.to("mps"), b.to("mps"), M.to("mps"))], 0.1, reg_m, 1e-7, 20000)
    assert_plans_close(plans[0], reference)


@mps
@pytest.mark.parametrize("n, m", [(500, 7), (3000, 40), (22725, 63), (300, 255), (1000, 1)])
def test_metal_matches_pot(n: int, m: int):
    a, b, M = assignment_problem(n, m, seed=n + m, device="mps")
    reference, n_iter_pot = pot(a, b, M)
    plans, n_iter = sinkhorn_unbalanced_ti_batch([(a, b, M)], 0.1, (1.0, 1.0), 1e-3, 1000)
    assert n_iter[0] == n_iter_pot
    assert_plans_close(plans[0], reference)
    assert torch.equal(assign(plans[0]), assign(reference))


@mps
def test_metal_batch_of_different_shapes():
    """A batch mixes problems of different sizes (one threadgroup never spans two problems) and the problems with more
    than 256 columns, which are solved with the PyTorch implementation."""
    problems = [
        assignment_problem(n, m, seed=i, device="mps")
        for i, (n, m) in enumerate([(700, 5), (257, 33), (5000, 80), (300, 300), (1, 3)])
    ]
    plans, n_iter = sinkhorn_unbalanced_ti_batch(problems, 0.1, (1.0, 1.0), 1e-3, 1000)
    for (a, b, M), plan, it in zip(problems, plans, n_iter):
        reference, n_iter_pot = pot(a, b, M)
        assert plan.shape == M.shape
        assert it == n_iter_pot
        assert_plans_close(plan, reference)


@mps
def test_metal_numerical_error_keeps_the_previous_iterate():
    """A row with a cost that underflows exp(-M / reg) makes K v zero: as POT, stop and keep the previous iterate (here
    the initial one, the plan is K itself)."""
    a, b, M = assignment_problem(400, 6, seed=3, device="mps")
    M[5] = 1e4
    reference, n_iter_pot = pot(a, b, M)
    plans, n_iter = sinkhorn_unbalanced_ti_batch([(a, b, M)], 0.1, (1.0, 1.0), 1e-3, 1000)
    assert n_iter[0] == n_iter_pot == 0
    assert_plans_close(plans[0], reference)


@mps
def test_metal_detects_nan():
    """A negative background supply makes v NaN in the first iteration: stop as POT does (the fast math of the Metal
    compiler must not optimize the check away)."""
    a, b, M = assignment_problem(400, 6, seed=4, device="mps")
    b[-1] = -10.0
    reference, n_iter_pot = pot(a, b, M)
    plans, n_iter = sinkhorn_unbalanced_ti_batch([(a, b, M)], 0.1, (1.0, 1.0), 1e-3, 1000)
    assert n_iter[0] == n_iter_pot == 0
    # POT builds K from log(a b^T), which is NaN in the column with the negative supply: compare the others
    assert_plans_close(plans[0][:, :-1], reference[:, :-1])


def test_single_problem_api_on_cpu():
    a, b, M = assignment_problem(200, 4, seed=5)
    plan = sinkhorn_unbalanced_ti(a, b, M, reg=0.1, reg_m=(1.0, 1.0), stop_thr=1e-3, max_iter=1000)
    assert_plans_close(plan, pot(a, b, M)[0])
