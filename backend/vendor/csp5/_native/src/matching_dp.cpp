#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static int match_single(
    const double *pred_vals,
    int n_pred,
    const double *obs_vals,
    int n_obs,
    double dummy_cost,
    const double *row_penalties,
    int *out_rows,
    int *out_cols,
    int out_cap) {
  if (!pred_vals || !obs_vals || !out_rows || !out_cols) {
    return 0;
  }
  if (n_pred <= 0 || n_obs <= 0) {
    return 0;
  }
  const int max_pairs = std::min(n_pred, n_obs);
  if (out_cap < max_pairs) {
    return 0;
  }

  std::vector<int> pred_order(n_pred);
  std::vector<int> obs_order(n_obs);
  std::iota(pred_order.begin(), pred_order.end(), 0);
  std::iota(obs_order.begin(), obs_order.end(), 0);

  std::stable_sort(pred_order.begin(), pred_order.end(),
                   [&](int a, int b) { return pred_vals[a] < pred_vals[b]; });
  std::stable_sort(obs_order.begin(), obs_order.end(),
                   [&](int a, int b) { return obs_vals[a] < obs_vals[b]; });

  std::vector<double> pred_sorted(n_pred);
  std::vector<double> obs_sorted(n_obs);
  for (int i = 0; i < n_pred; ++i) {
    pred_sorted[i] = pred_vals[pred_order[i]];
  }
  for (int j = 0; j < n_obs; ++j) {
    obs_sorted[j] = obs_vals[obs_order[j]];
  }

  std::vector<double> penalty_sorted;
  if (row_penalties) {
    penalty_sorted.resize(n_pred);
    for (int i = 0; i < n_pred; ++i) {
      penalty_sorted[i] = row_penalties[pred_order[i]];
    }
  }

  const int n_obs_p1 = n_obs + 1;
  const int n_pred_p1 = n_pred + 1;
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> dp(static_cast<size_t>(n_pred_p1) * n_obs_p1, inf);
  std::vector<int8_t> back(static_cast<size_t>(n_pred_p1) * n_obs_p1, -1);

  auto idx = [n_obs_p1](int i, int j) { return i * n_obs_p1 + j; };

  dp[idx(0, 0)] = 0.0;
  const bool allow_skip_pred = n_pred > n_obs;
  const bool allow_skip_obs = n_obs > n_pred;
  if (allow_skip_pred) {
    for (int i = 1; i <= n_pred; ++i) {
      dp[idx(i, 0)] = dp[idx(i - 1, 0)] + dummy_cost;
      back[idx(i, 0)] = 1;
    }
  }
  if (allow_skip_obs) {
    for (int j = 1; j <= n_obs; ++j) {
      dp[idx(0, j)] = dp[idx(0, j - 1)] + dummy_cost;
      back[idx(0, j)] = 2;
    }
  }

  for (int i = 1; i <= n_pred; ++i) {
    const double pred_val = pred_sorted[i - 1];
    const double penalty = row_penalties ? penalty_sorted[i - 1] : 0.0;
    for (int j = 1; j <= n_obs; ++j) {
      double best_cost = dp[idx(i - 1, j - 1)] + std::abs(pred_val - obs_sorted[j - 1]) + penalty;
      int8_t best_move = 0;
      if (allow_skip_pred) {
        double skip_pred = dp[idx(i - 1, j)] + dummy_cost;
        if (skip_pred < best_cost) {
          best_cost = skip_pred;
          best_move = 1;
        }
      }
      if (allow_skip_obs) {
        double skip_obs = dp[idx(i, j - 1)] + dummy_cost;
        if (skip_obs < best_cost) {
          best_cost = skip_obs;
          best_move = 2;
        }
      }
      dp[idx(i, j)] = best_cost;
      back[idx(i, j)] = best_move;
    }
  }

  std::vector<int> rows;
  std::vector<int> cols;
  rows.reserve(max_pairs);
  cols.reserve(max_pairs);

  int i = n_pred;
  int j = n_obs;
  while (i > 0 || j > 0) {
    int8_t move = back[idx(i, j)];
    if (move == 0) {
      rows.push_back(pred_order[i - 1]);
      cols.push_back(obs_order[j - 1]);
      --i;
      --j;
    } else if (move == 1) {
      --i;
    } else {
      --j;
    }
  }

  const int count = static_cast<int>(rows.size());
  for (int k = 0; k < count; ++k) {
    const int out_idx = count - 1 - k;
    out_rows[k] = rows[out_idx];
    out_cols[k] = cols[out_idx];
  }
  return count;
}

extern "C" int nmrexp_match_indices_dp(
    const double *pred_vals,
    int n_pred,
    const double *obs_vals,
    int n_obs,
    double dummy_cost,
    const double *row_penalties,
    int *out_rows,
    int *out_cols,
    int out_cap) {
  return match_single(
      pred_vals,
      n_pred,
      obs_vals,
      n_obs,
      dummy_cost,
      row_penalties,
      out_rows,
      out_cols,
      out_cap);
}

extern "C" int nmrexp_match_indices_dp_batch(
    const double **pred_vals,
    const int *n_pred,
    const double **obs_vals,
    const int *n_obs,
    const double *dummy_costs,
    const double **row_penalties,
    int **out_rows,
    int **out_cols,
    int *out_counts,
    const int *out_caps,
    int batch_size,
    int n_threads) {
  if (!pred_vals || !n_pred || !obs_vals || !n_obs || !dummy_costs ||
      !out_rows || !out_cols || !out_counts || !out_caps) {
    return 0;
  }
  if (batch_size <= 0) {
    return 0;
  }
#ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
#endif
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < batch_size; ++i) {
    const double *pred_ptr = pred_vals[i];
    const double *obs_ptr = obs_vals[i];
    const double *penalties = row_penalties ? row_penalties[i] : nullptr;
    const double dummy_cost = dummy_costs[i];
    out_counts[i] = match_single(
        pred_ptr,
        n_pred[i],
        obs_ptr,
        n_obs[i],
        dummy_cost,
        penalties,
        out_rows[i],
        out_cols[i],
        out_caps[i]);
  }
  return batch_size;
}
