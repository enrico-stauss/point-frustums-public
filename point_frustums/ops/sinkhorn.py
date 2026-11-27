"""
Translation invariant unbalanced Sinkhorn with KL marginal penalties, as used by the optimal transport target assignment.

The iteration is the H-Sinkhorn of Séjourné et al. (2022) with the reference measure a b^T: each half step maximizes
the translation invariant dual exactly. For rho_1 == rho_2 it is identical to POT's
ot.unbalanced.sinkhorn_unbalanced_translation_invariant (reg_type="kl", the plans agree up to float32 round-off). For
rho_1 != rho_2, POT's iteration does not converge to the optimum, this one does (see ReadMe.md). POT also evaluates the
stopping criterion and checks for numerical errors on the host in every iteration, which synchronizes with the GPU
several times per iteration. Here:

- MPS: Metal kernels (csrc/sinkhorn/sinkhorn.metal), 5 launches per iteration for all problems of a batch together.
  The stopping criterion stays on the device and is checked by the host only every `check_every` iterations.
- Other devices (and problems with more than 256 columns): the same iteration in PyTorch, which keeps the stopping
  criterion on the device and synchronizes only every `check_every` iterations. A problem that has converged is
  frozen, so the result is the same as with a check in every iteration.

The derivation of the iteration and how it maps to the code: csrc/sinkhorn/ReadMe.md.
"""

from collections.abc import Sequence
from functools import cache
from pathlib import Path

import torch

_SOURCE = Path(__file__).parent / "csrc" / "sinkhorn" / "sinkhorn.metal"
# Must match SINKHORN_THREADS, SINKHORN_MAX_COLS and SINKHORN_BLOCK_ROWS of the kernels
_THREADS = 256
_MAX_COLS = 256
_BLOCK_ROWS = 256


def _parameters(reg: float, reg_m: tuple[float, float]) -> dict[str, float]:
    """The exponents of the iteration (notation of csrc/sinkhorn/ReadMe.md), for finite penalties rho_1, rho_2."""
    r1, r2 = reg_m
    if not (0 < r1 < float("inf") and 0 < r2 < float("inf") and reg > 0):
        raise ValueError(f"Expected finite positive reg and reg_m, got reg={reg}, reg_m={reg_m}")
    return {
        "reg": reg,
        "fi_1": r1 / (r1 + reg),
        "fi_2": r2 / (r2 + reg),
        "theta_1": r1 / (reg + r1 + r2),
        "theta_2": r2 / (reg + r1 + r2),
        "ratio_1": reg / r1,
        "ratio_2": reg / r2,
        "fi_12": r1 * r2 / (r1 + r2),
    }


@cache
def _library():
    return torch.mps.compile_shader("#include <metal_stdlib>\nusing namespace metal;\n" + _SOURCE.read_text())


def _sinkhorn_metal(
    problems: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    reg: float,
    reg_m: tuple[float, float],
    stop_thr: float,
    max_iter: int,
    check_every: int = 5,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    device = problems[0][2].device
    params = torch.tensor([*_parameters(reg, reg_m).values(), stop_thr, max_iter], dtype=torch.float32, device=device)
    a = torch.cat([p[0].reshape(-1) for p in problems]).float().contiguous()
    b = torch.cat([p[1].reshape(-1) for p in problems]).float().contiguous()
    K = torch.cat([p[2].reshape(-1) for p in problems]).float().contiguous()  # the cost, overwritten by the plan
    shapes = [tuple(p[2].shape) for p in problems]
    info, blocks, row, col, mat = [], [], 0, 0, 0
    for index, (n, c) in enumerate(shapes):
        starts = range(0, n, _BLOCK_ROWS)
        info.extend([n, c, row, col, mat, len(blocks) // 3, len(starts)])
        for start in starts:
            blocks.extend([index, start, min(start + _BLOCK_ROWS, n)])
        row, col, mat = row + n, col + c, mat + n * c
    if mat >= 2**31:
        raise ValueError("The Metal Sinkhorn uses 32 bit indices, the problems are too large")
    info = torch.tensor(info, dtype=torch.int32, device=device)
    blocks = torch.tensor(blocks, dtype=torch.int32, device=device)
    n_problems, n_blocks = len(problems), blocks.numel() // 3

    u, u_bar = torch.empty_like(a), torch.empty_like(a)
    v, v_bar = torch.empty_like(b), torch.empty_like(b)
    state = torch.empty((n_problems, 4), dtype=torch.float32, device=device)
    status = torch.empty((n_problems,), dtype=torch.int32, device=device)
    n_iter = torch.empty((n_problems,), dtype=torch.int32, device=device)
    partial_1 = torch.empty((n_blocks,), dtype=torch.float32, device=device)
    partial_2 = torch.empty((n_blocks, 2), dtype=torch.float32, device=device)
    partial_cols = torch.empty((n_blocks, _MAX_COLS), dtype=torch.float32, device=device)

    lib = _library()
    rows, cols = (
        {"threads": n_blocks * _THREADS, "group_size": _THREADS},
        {"threads": n_problems * _THREADS, "group_size": _THREADS},
    )
    lib.sinkhorn_init_rows(a, b, K, u, u_bar, info, blocks, params, **rows)
    lib.sinkhorn_init_columns(b, v, v_bar, state, status, n_iter, info, params, **cols)
    for i in range(max_iter):
        lib.sinkhorn_rows_u_hat(a, K, u_bar, v_bar, status, partial_1, info, blocks, params, **rows)
        lib.sinkhorn_rows_ktu(
            a, K, u_bar, state, status, partial_1, partial_2, partial_cols, info, blocks, params, **rows
        )
        lib.sinkhorn_columns(b, v, v_bar, state, status, partial_2, partial_cols, info, params, **cols)
        lib.sinkhorn_rows_translate(u, u_bar, state, status, partial_2, info, blocks, **rows)
        lib.sinkhorn_finalize(state, status, n_iter, partial_2, info, params, **cols)
        # The state stays on the device, kernels of problems that are done return immediately: only check sometimes
        if (i + 1) % check_every == 0 and not bool((status == 0).any()):
            break
    lib.sinkhorn_plan(K, u, v, info, blocks, **rows)
    plans = [chunk.view(n, c) for chunk, (n, c) in zip(K.split([n * c for n, c in shapes]), shapes)]
    return plans, n_iter


def _sinkhorn_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    M: torch.Tensor,
    reg: float,
    reg_m: tuple[float, float],
    stop_thr: float,
    max_iter: int,
    check_every: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    p = _parameters(reg, reg_m)
    K = a[:, None] * b[None, :] * torch.exp(-M / reg)
    u, v = torch.ones_like(a), torch.ones_like(b)
    u_bar, v_bar = u, v
    done = torch.zeros((), dtype=torch.bool, device=M.device)
    n_iter = torch.zeros((), dtype=torch.int32, device=M.device)
    for i in range(max_iter):
        # The exact maximization of the translation invariant dual over u_bar, then over v_bar (ReadMe.md)
        u_hat = (a / (K @ v_bar)) ** p["fi_1"]
        u_bar_new = (
            u_hat * (torch.sum(b * v_bar ** (-p["ratio_2"])) / torch.sum(a * u_hat ** (-p["ratio_1"]))) ** p["theta_2"]
        )
        ktu = K.T @ u_bar_new
        sum_u = torch.sum(a * u_bar_new ** (-p["ratio_1"]))
        v_hat = (b / ktu) ** p["fi_2"]
        v_bar_new = v_hat * (sum_u / torch.sum(b * v_hat ** (-p["ratio_2"]))) ** p["theta_1"]
        error = torch.any(ktu == 0) | ~torch.all(torch.isfinite(u_bar_new)) | ~torch.all(torch.isfinite(v_bar_new))
        t = (sum_u / torch.sum(b * v_bar_new ** (-p["ratio_2"]))) ** (p["fi_12"] / reg)
        u_new, v_new = u_bar_new * t, v_bar_new / t
        err_u = (u_new - u).abs().max() / torch.maximum(torch.maximum(u_new.abs().max(), u.abs().max()), u.new_ones(()))
        err_v = (v_new - v).abs().max() / torch.maximum(torch.maximum(v_new.abs().max(), v.abs().max()), v.new_ones(()))
        # A numerical error keeps the previous iterate, convergence keeps the new one; both end the iteration
        keep = done | error
        u, v = torch.where(keep, u, u_new), torch.where(keep, v, v_new)
        u_bar, v_bar = torch.where(keep, u_bar, u_bar_new), torch.where(keep, v_bar, v_bar_new)
        n_iter = n_iter + (~keep).int()
        done = keep | (0.5 * (err_u + err_v) < stop_thr)
        if (i + 1) % check_every == 0 and bool(done):
            break
    return u[:, None] * K * v[None, :], n_iter


def sinkhorn_unbalanced_ti_batch(
    problems: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    reg: float,
    reg_m: tuple[float, float] = (1.0, 1.0),
    stop_thr: float = 1e-6,
    max_iter: int = 1000,
) -> tuple[list[torch.Tensor], list[int]]:
    """
    Solve several unbalanced OT problems (a: (N,), b: (C,), M: (N, C)), with the same hyperparameters.
    :param problems: The marginals a, b and the cost M of each problem
    :param reg: The entropic regularization epsilon
    :param reg_m: The KL penalties rho_1, rho_2 of the marginals a and b
    :param stop_thr: Stop when the relative change of the scalings u, v is below this threshold
    :param max_iter: The maximum number of iterations
    :return: The transport plans (N, C) and the number of iterations of each problem
    """
    if not problems:
        return [], []
    device = problems[0][2].device
    metal = [device.type == "mps" and 0 < M.shape[1] <= _MAX_COLS and M.shape[0] > 0 for _, _, M in problems]
    plans: list = [None] * len(problems)
    n_iter: list = [None] * len(problems)
    if any(metal):
        index = [i for i, m in enumerate(metal) if m]
        result, iters = _sinkhorn_metal([problems[i] for i in index], reg, tuple(reg_m), stop_thr, max_iter)
        for i, plan, it in zip(index, result, iters.tolist()):
            plans[i], n_iter[i] = plan, it
    for i, m in enumerate(metal):
        if not m:
            a, b, M = problems[i]
            plans[i], it = _sinkhorn_torch(a.float(), b.float(), M.float(), reg, tuple(reg_m), stop_thr, max_iter)
            n_iter[i] = int(it)
    return plans, n_iter


def sinkhorn_unbalanced_ti(
    a: torch.Tensor,
    b: torch.Tensor,
    M: torch.Tensor,
    reg: float,
    reg_m: tuple[float, float] = (1.0, 1.0),
    stop_thr: float = 1e-6,
    max_iter: int = 1000,
) -> torch.Tensor:
    """The transport plan (N, C) of a single problem, see sinkhorn_unbalanced_ti_batch."""
    plans, _ = sinkhorn_unbalanced_ti_batch([(a, b, M)], reg, reg_m, stop_thr, max_iter)
    return plans[0]
