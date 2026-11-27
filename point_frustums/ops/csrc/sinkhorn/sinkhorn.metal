// Translation invariant unbalanced Sinkhorn (KL marginal penalties, the H-Sinkhorn of Séjourné et al. 2022) for a
// batch of problems.
// See ReadMe.md in this directory for the derivation and how the code maps to it.
//
// Problem p has the marginals a_p (N_p), b_p (C_p) and the cost/kernel matrix K_p (N_p, C_p), all problems are
// concatenated. The rows of every problem are split into blocks of SINKHORN_BLOCK_ROWS rows, one threadgroup of
// SINKHORN_THREADS threads per block. Inside a block, the rows are distributed over the simdgroups and the columns over
// the lanes (C_p <= SINKHORN_MAX_COLS), so that reading a row is coalesced and both K v (a sum along the row,
// simd_sum) and K^T u (a sum along the column, accumulated per lane in registers) come from the same read of the row.
// Sums over all rows of a problem are written as one partial per block and reduced by the next kernel, which makes
// the result deterministic (no atomics).
//
// One iteration consists of 5 launches (rows_u_hat, rows_ktu, columns, rows_translate, finalize). The state of each
// problem (running/converged/numerical error) stays on the device: the kernels of a problem that is not running
// return immediately, so the host only has to check it every few iterations.

#define SINKHORN_SIMDS 8
#define SINKHORN_THREADS (32 * SINKHORN_SIMDS)
#define SINKHORN_COLS_PER_LANE 8
#define SINKHORN_MAX_COLS (32 * SINKHORN_COLS_PER_LANE)
#define SINKHORN_BLOCK_ROWS 256  // == SINKHORN_THREADS == 32 * SINKHORN_SIMDS

#define RUNNING 0
#define CONVERGED 1
#define NUMERICAL_ERROR 2

// The parameters, in this order (see sinkhorn.py: _parameters); the notation follows ReadMe.md
struct SinkhornParams {
  float reg;      // epsilon, the entropic regularization
  float fi_1;     // rho_1 / (rho_1 + eps)
  float fi_2;     // rho_2 / (rho_2 + eps)
  float theta_1;    // rho_1 / (eps + rho_1 + rho_2)
  float theta_2;    // rho_2 / (eps + rho_1 + rho_2)
  float ratio_1;  // eps / rho_1
  float ratio_2;  // eps / rho_2
  float fi_12;    // rho_1 rho_2 / (rho_1 + rho_2), the scale of the translation
  float stop_thr;
  float max_iter;
};

// Per problem: N, C, offset of a/u (rows), offset of b/v (columns), offset of K, first block, number of blocks
#define INFO_SIZE 7
// Per block: problem, first row, end row (relative to the problem)
#define BLOCK_SIZE 3
// Per problem float state: factor of the update of u, translation t, max|v - v_prev|, max(|v|, |v_prev|)
#define STATE_SIZE 4

// 32 bit indices: 64 bit integer arithmetic is slow on the GPU (the host checks that all sizes fit)
struct Problem {
  int n, c, row, col, mat, first_block, n_blocks;
};

inline Problem problem_info(device const int* info, int p) {
  device const int* x = info + INFO_SIZE * p;
  return Problem{x[0], x[1], x[2], x[3], x[4], x[5], x[6]};
}

// Sum (or max) over the threadgroup, returned to all threads. scratch: SINKHORN_SIMDS floats.
inline float tg_sum(float x, threadgroup float* scratch, uint sg, uint lane) {
  x = simd_sum(x);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) scratch[sg] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return simd_sum(lane < SINKHORN_SIMDS ? scratch[lane] : 0.0f);
}

inline float tg_max(float x, threadgroup float* scratch, uint sg, uint lane) {
  x = simd_max(x);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) scratch[sg] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return simd_max(lane < SINKHORN_SIMDS ? scratch[lane] : -INFINITY);
}

// NaN or +-inf from the bit pattern: with fast math (the default of the Metal compiler), isnan and isinf may be
// optimized away because the compiler assumes that there are none
inline bool not_finite(float x) { return (as_type<uint>(x) & 0x7f800000u) == 0x7f800000u; }

// K = a b^T exp(-M / eps) in place of M, u = u_bar = 1 (per block)
kernel void sinkhorn_init_rows(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* K [[buffer(2)]],
    device float* u [[buffer(3)]],
    device float* u_bar [[buffer(4)]],
    device const int* info [[buffer(5)]],
    device const int* blocks [[buffer(6)]],
    device const SinkhornParams& p [[buffer(7)]],
    uint blk [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const Problem P = problem_info(info, blocks[BLOCK_SIZE * blk]);
  for (int i = blocks[BLOCK_SIZE * blk + 1] + sg; i < blocks[BLOCK_SIZE * blk + 2]; i += SINKHORN_SIMDS) {
    device float* row = K + P.mat + i * P.c;
    const float ai = a[P.row + i];
    for (int j = lane; j < P.c; j += 32) row[j] = ai * b[P.col + j] * exp(-row[j] / p.reg);
    if (lane == 0) {
      u[P.row + i] = 1.0f;
      u_bar[P.row + i] = 1.0f;
    }
  }
}

// v = v_bar = 1, the factor of the first update of u, the state (per problem)
kernel void sinkhorn_init_columns(
    device const float* b [[buffer(0)]],
    device float* v [[buffer(1)]],
    device float* v_bar [[buffer(2)]],
    device float* state [[buffer(3)]],
    device int* status [[buffer(4)]],
    device int* n_iter [[buffer(5)]],
    device const int* info [[buffer(6)]],
    device const SinkhornParams& p [[buffer(7)]],
    uint prob [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[SINKHORN_SIMDS];
  const Problem P = problem_info(info, prob);
  float s = 0.0f;
  for (int j = tid; j < P.c; j += SINKHORN_THREADS) {
    v[P.col + j] = 1.0f;
    v_bar[P.col + j] = 1.0f;
    s += b[P.col + j];  // B = sum(b v_bar^(-eps/rho_2)) with v_bar = 1
  }
  s = tg_sum(s, scratch, sg, lane);
  if (tid == 0) {
    state[STATE_SIZE * prob + 0] = pow(s, p.theta_2);
    status[prob] = RUNNING;
    n_iter[prob] = 0;
  }
}

// Pass 1 over the rows: u_hat = (a / K v_bar)^fi_1 (stored in u_bar), partial A_hat = sum(a u_hat^(-eps/rho_1))
kernel void sinkhorn_rows_u_hat(
    device const float* a [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device float* u_bar [[buffer(2)]],
    device const float* v_bar [[buffer(3)]],
    device const int* status [[buffer(4)]],
    device float* partial [[buffer(5)]],
    device const int* info [[buffer(6)]],
    device const int* blocks [[buffer(7)]],
    device const SinkhornParams& p [[buffer(8)]],
    uint blk [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[SINKHORN_SIMDS];
  threadgroup float vb[SINKHORN_MAX_COLS];
  const int prob = blocks[BLOCK_SIZE * blk];
  if (status[prob] != RUNNING) return;
  const Problem P = problem_info(info, prob);
  for (int j = tid; j < P.c; j += SINKHORN_THREADS) vb[j] = v_bar[P.col + j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // One thread per row (SINKHORN_BLOCK_ROWS == SINKHORN_THREADS): the dot product and the powers of all rows are
  // evaluated in parallel, the lanes read vb[j] at the same time (a broadcast from threadgroup memory)
  float s = 0.0f;
  const int i = blocks[BLOCK_SIZE * blk + 1] + int(tid);
  if (i < blocks[BLOCK_SIZE * blk + 2]) {
    device const float* row = K + P.mat + i * P.c;
    float kv = 0.0f;
    for (int j = 0; j < P.c; ++j) kv += row[j] * vb[j];
    const float ai = a[P.row + i];
    const float u_hat = pow(ai / kv, p.fi_1);
    u_bar[P.row + i] = u_hat;
    s = ai * pow(u_hat, -p.ratio_1);
  }
  s = tg_sum(s, scratch, sg, lane);
  if (tid == 0) partial[blk] = s;
}

// Pass 2 over the rows: u_bar = u_hat * norm, partials of K^T u_bar, sum(a u_bar^(-eps/rho_1)) and of the error flag
kernel void sinkhorn_rows_ktu(
    device const float* a [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device float* u_bar [[buffer(2)]],
    device const float* state [[buffer(3)]],
    device const int* status [[buffer(4)]],
    device const float* partial_in [[buffer(5)]],
    device float* partial_out [[buffer(6)]],  // per block: sum, error flag
    device float* partial_cols [[buffer(7)]],  // per block: SINKHORN_MAX_COLS
    device const int* info [[buffer(8)]],
    device const int* blocks [[buffer(9)]],
    device const SinkhornParams& p [[buffer(10)]],
    uint blk [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[SINKHORN_SIMDS];
  threadgroup float cols[SINKHORN_SIMDS][32];
  const int prob = blocks[BLOCK_SIZE * blk];
  if (status[prob] != RUNNING) return;
  const Problem P = problem_info(info, prob);
  // The normalization of u from the partials of pass 1 of all blocks of the problem
  float s = 0.0f;
  for (int k = tid; k < P.n_blocks; k += SINKHORN_THREADS) s += partial_in[P.first_block + k];
  // A_hat^(-theta_2) B^theta_2 (B^theta_2 from the previous columns step)
  const float norm = pow(tg_sum(s, scratch, sg, lane), -p.theta_2) * state[STATE_SIZE * prob + 0];

  float acc[SINKHORN_COLS_PER_LANE];
  for (int k = 0; k < SINKHORN_COLS_PER_LANE; ++k) acc[k] = 0.0f;
  s = 0.0f;
  bool error = false;
  // Each simdgroup takes 32 consecutive rows (SINKHORN_BLOCK_ROWS == 32 * SINKHORN_SIMDS): lane r scales u_bar of
  // row r (in parallel), then the rows are accumulated one by one with the columns distributed over the lanes
  const int row0 = blocks[BLOCK_SIZE * blk + 1] + 32 * int(sg);
  const int n_rows = clamp(blocks[BLOCK_SIZE * blk + 2] - row0, 0, 32);
  float ub = 0.0f;
  if (int(lane) < n_rows) {
    const int i = row0 + int(lane);
    ub = u_bar[P.row + i] * norm;
    u_bar[P.row + i] = ub;
    s = a[P.row + i] * pow(ub, -p.ratio_1);
    error = not_finite(ub);
  }
  for (int r = 0; r < n_rows; ++r) {
    device const float* row = K + P.mat + (row0 + r) * P.c;
    const float ur = simd_broadcast(ub, ushort(r));
    // Constant trip count: the loop is unrolled and acc stays in registers
    for (int k = 0; k < SINKHORN_COLS_PER_LANE; ++k) {
      const int j = lane + 32 * k;
      if (j < P.c) acc[k] += row[j] * ur;
    }
  }
  s = tg_sum(s, scratch, sg, lane);
  const float e = tg_max(error ? 1.0f : 0.0f, scratch, sg, lane);
  if (tid == 0) {
    partial_out[2 * blk + 0] = s;
    partial_out[2 * blk + 1] = e;
  }
  // Reduce the per lane column accumulators over the simdgroups, 32 columns per round
  for (int k = 0; k < SINKHORN_COLS_PER_LANE; ++k) {
    if (32 * k >= P.c) break;  // uniform over the threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);
    cols[sg][lane] = acc[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && 32 * k + lane < P.c) {
      float x = 0.0f;
      for (int r = 0; r < SINKHORN_SIMDS; ++r) x += cols[r][lane];
      partial_cols[SINKHORN_MAX_COLS * blk + 32 * k + lane] = x;
    }
  }
}

// The columns (per problem): K^T u_bar, v_bar, the translation t, v and the change of v
kernel void sinkhorn_columns(
    device const float* b [[buffer(0)]],
    device float* v [[buffer(1)]],
    device float* v_bar [[buffer(2)]],
    device float* state [[buffer(3)]],
    device int* status [[buffer(4)]],
    device const float* partial [[buffer(5)]],  // per block: sum, error flag
    device const float* partial_cols [[buffer(6)]],
    device const int* info [[buffer(7)]],
    device const SinkhornParams& p [[buffer(8)]],
    uint prob [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[SINKHORN_SIMDS];
  if (status[prob] != RUNNING) return;
  const Problem P = problem_info(info, prob);
  float s = 0.0f, e = 0.0f;
  for (int k = tid; k < P.n_blocks; k += SINKHORN_THREADS) {
    s += partial[2 * (P.first_block + k) + 0];
    e = max(e, partial[2 * (P.first_block + k) + 1]);
  }
  const float sum_u = tg_sum(s, scratch, sg, lane);
  bool error = tg_max(e, scratch, sg, lane) > 0.0f;

  // C <= SINKHORN_MAX_COLS == SINKHORN_THREADS: at most one column per thread
  const int j = tid;
  const bool active = j < P.c;
  float bj = 0.0f, v_hat = 0.0f;
  if (active) {
    float ktu = 0.0f;
    for (int k = 0; k < P.n_blocks; ++k) ktu += partial_cols[SINKHORN_MAX_COLS * (P.first_block + k) + j];
    bj = b[P.col + j];
    error = error || ktu == 0.0f;
    v_hat = pow(bj / ktu, p.fi_2);
  }
  const float norm = pow(tg_sum(active ? bj * pow(v_hat, -p.ratio_2) : 0.0f, scratch, sg, lane), -p.theta_1) * pow(sum_u, p.theta_1);  // B_hat^(-theta_1) A^theta_1
  const float vb = v_hat * norm;
  error = error || (active && not_finite(vb));
  const float sum_v = tg_sum(active ? bj * pow(vb, -p.ratio_2) : 0.0f, scratch, sg, lane);
  if (tg_max(error ? 1.0f : 0.0f, scratch, sg, lane) > 0.0f) {
    // Keep u and v of the previous iteration (as POT, which warns and stops)
    if (tid == 0) status[prob] = NUMERICAL_ERROR;
    return;
  }
  const float t = pow(sum_u / sum_v, p.fi_12 / p.reg);
  float dv = 0.0f, mv = 0.0f;
  if (active) {
    const float v_old = v[P.col + j], v_new = vb / t;
    dv = fabs(v_new - v_old);
    mv = max(fabs(v_new), fabs(v_old));
    v[P.col + j] = v_new;
    v_bar[P.col + j] = vb;
  }
  dv = tg_max(dv, scratch, sg, lane);
  mv = tg_max(mv, scratch, sg, lane);
  if (tid == 0) {
    state[STATE_SIZE * prob + 0] = pow(sum_v, p.theta_2);  // B^theta_2, the factor of the next update of u
    state[STATE_SIZE * prob + 1] = t;
    state[STATE_SIZE * prob + 2] = dv;
    state[STATE_SIZE * prob + 3] = mv;
  }
}

// Pass 3 over the rows: u = u_bar * t and the partial change of u
kernel void sinkhorn_rows_translate(
    device float* u [[buffer(0)]],
    device const float* u_bar [[buffer(1)]],
    device const float* state [[buffer(2)]],
    device const int* status [[buffer(3)]],
    device float* partial [[buffer(4)]],  // per block: max|u - u_prev|, max(|u|, |u_prev|)
    device const int* info [[buffer(5)]],
    device const int* blocks [[buffer(6)]],
    uint blk [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[SINKHORN_SIMDS];
  const int prob = blocks[BLOCK_SIZE * blk];
  if (status[prob] != RUNNING) return;
  const Problem P = problem_info(info, prob);
  const float t = state[STATE_SIZE * prob + 1];
  float du = 0.0f, mu = 0.0f;
  for (int i = blocks[BLOCK_SIZE * blk + 1] + tid; i < blocks[BLOCK_SIZE * blk + 2]; i += SINKHORN_THREADS) {
    const float u_old = u[P.row + i], u_new = u_bar[P.row + i] * t;
    du = max(du, fabs(u_new - u_old));
    mu = max(mu, max(fabs(u_new), fabs(u_old)));
    u[P.row + i] = u_new;
  }
  du = tg_max(du, scratch, sg, lane);
  mu = tg_max(mu, scratch, sg, lane);
  if (tid == 0) {
    partial[2 * blk + 0] = du;
    partial[2 * blk + 1] = mu;
  }
}

// The stopping criterion (per problem)
kernel void sinkhorn_finalize(
    device const float* state [[buffer(0)]],
    device int* status [[buffer(1)]],
    device int* n_iter [[buffer(2)]],
    device const float* partial [[buffer(3)]],
    device const int* info [[buffer(4)]],
    device const SinkhornParams& p [[buffer(5)]],
    uint prob [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[SINKHORN_SIMDS];
  if (status[prob] != RUNNING) return;
  const Problem P = problem_info(info, prob);
  float du = 0.0f, mu = 0.0f;
  for (int k = tid; k < P.n_blocks; k += SINKHORN_THREADS) {
    du = max(du, partial[2 * (P.first_block + k) + 0]);
    mu = max(mu, partial[2 * (P.first_block + k) + 1]);
  }
  du = tg_max(du, scratch, sg, lane);
  mu = tg_max(mu, scratch, sg, lane);
  if (tid == 0) {
    const float dv = state[STATE_SIZE * prob + 2], mv = state[STATE_SIZE * prob + 3];
    const float err = 0.5f * (du / max(mu, 1.0f) + dv / max(mv, 1.0f));
    n_iter[prob] += 1;
    if (err < p.stop_thr) status[prob] = CONVERGED;
  }
}

// The plan u_i K_ij v_j in place of K (per block)
kernel void sinkhorn_plan(
    device float* K [[buffer(0)]],
    device const float* u [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const int* info [[buffer(3)]],
    device const int* blocks [[buffer(4)]],
    uint blk [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const Problem P = problem_info(info, blocks[BLOCK_SIZE * blk]);
  for (int i = blocks[BLOCK_SIZE * blk + 1] + sg; i < blocks[BLOCK_SIZE * blk + 2]; i += SINKHORN_SIMDS) {
    device float* row = K + P.mat + i * P.c;
    const float ui = u[P.row + i];
    for (int j = lane; j < P.c; j += 32) row[j] = ui * row[j] * v[P.col + j];
  }
}
