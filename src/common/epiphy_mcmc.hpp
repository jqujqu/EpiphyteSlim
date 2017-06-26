/* Copyright (C) 2015-16 University of Southern California and
 *                       Andrew D. Smith and Jenny Qu
 *
 * Authors: Jenny Qu and Andrew Smith
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */

#ifndef EPIPHY_MCMC_HPP
#define EPIPHY_MCMC_HPP

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//////////////   MCMC sampling using Markov Blancket    ////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#include <vector>
#include <random>
#include <iostream>


#include "epiphy_utils.hpp"
#include "param_set.hpp"
#include "sufficient_statistics_helpers.hpp"


class epiphy_mcmc {
public:
  epiphy_mcmc(const size_t s) :
    the_seed(s) {
    dis = std::uniform_real_distribution<>(0, 1);
    //gen = std::mt19937_64(the_seed);
    gen = std::minstd_rand0(the_seed);
  }

  template <class T> void
  sample_states(const bool OBS,
                const std::vector<size_t> &subtree_sizes,
                const std::vector<size_t> &parent_ids,
                const param_set &params,
                const std::vector<std::vector<double> > &marginals,
                const std::vector<std::pair<size_t, size_t> > &blocks,
                std::vector<std::vector<T> > &states) const;

  template <class T> void
  sample_states(const bool OBS,
                const std::vector<size_t> &subtree_sizes,
                const std::vector<size_t> &parent_ids,
                const suff_stat &ss,
                const std::vector<std::vector<double> > &marginals,
                const std::vector<std::pair<size_t, size_t> > &blocks,
                std::vector<std::vector<T> > &states) const;

  template <class T> void
  sample_states_rev(const bool OBS,
                    const std::vector<size_t> &subtree_sizes,
                    const std::vector<size_t> &parent_ids,
                    const param_set &params,
                    const std::vector<std::vector<double> > &marginals,
                    const std::vector<std::pair<size_t, size_t> > &blocks,
                    std::vector<std::vector<T> > &states) const;

private:

  template <class T> void
  accept_or_reject_proposal(const double log_ratio, const size_t idx,
                            std::vector<T> &states) const;
  template <class T> void
  sample_states_middle(const std::vector<size_t> &subtree_sizes,
                       const std::vector<size_t> &parent_ids,
                       const pair_state &logG,
                       const std::vector<triple_state> &logGP,
                       const std::vector<double> &marginals,
                       const std::vector<T> &states_prev,
                       std::vector<T> &states_curr,
                       const std::vector<T> &states_next,
                       const size_t node_id) const;

  template <class T> void
  sample_states_end(const std::vector<size_t> &subtree_sizes,
                    const std::vector<size_t> &parent_ids,
                    const pair_state &logG,
                    const std::vector<triple_state> &logGP,
                    const std::vector<double> &marginals,
                    const std::vector<T> &states_prev,
                    std::vector<T> &states_curr,
                    const size_t node_id) const;

  template <class T> void
  sample_states_start(const std::vector<size_t> &subtree_sizes,
                      const std::vector<size_t> &parent_ids,
                      const std::pair<double, double> &log_pi,
                      const pair_state &logG,
                      const std::vector<pair_state> &logP,
                      const std::vector<triple_state> &logGP,
                      const std::vector<double> &marginals,
                      std::vector<T> &states_curr,
                      const std::vector<T> &states_next,
                      const size_t node_id) const;

  template <class T> void
  sample_states_indep(const std::vector<size_t> &subtree_sizes,
                      const std::vector<size_t> &parent_ids,
                      const std::pair<double, double> &log_pi,
                      const std::vector<pair_state> &logP,
                      const std::vector<double> &marginals,
                      std::vector<T> &states_curr,
                      const size_t node_id) const;

  mutable std::uniform_real_distribution<> dis;
  //mutable std::mt19937_64 gen;
  mutable std::minstd_rand0 gen;
  size_t the_seed;
};


/*
  prev/curr/next denote site positions in the genome
  parent/self/child denote species (nodes) in the phylogenetic tree
*/
template <class T>
static double
MB_state_middle(const std::vector<size_t> &subtree_sizes,
                const std::vector<size_t> &parent_ids,
                const pair_state &logG, // the matrix made of f0 & f1
                const std::vector<triple_state> &logGP,
                const std::vector<T> &meth_prev,
                const std::vector<T> &meth_curr,
                const std::vector<T> &meth_next,
                const size_t node_id) {

  const size_t parent = parent_ids[node_id];
  double lp0 = 0.0, lp1 = 0.0;

  // current and parent level
  if (is_root(node_id)) {
    const size_t prev_self_state = meth_prev[node_id];
    lp0 += logG(prev_self_state, 0);
    lp1 += logG(prev_self_state, 1);
    const size_t next_self_state = meth_next[node_id];
    lp0 += logG(0, next_self_state);
    lp1 += logG(1, next_self_state);
  } else {
    const size_t curr_parent_state = meth_curr[parent];
    const size_t prev_self_state = meth_prev[node_id];
    lp0 += logGP[node_id](prev_self_state, curr_parent_state, 0);
    lp1 += logGP[node_id](prev_self_state, curr_parent_state, 1);
    const size_t next_parent_state = meth_next[parent];
    const size_t next_self_state = meth_next[node_id];
    lp0 += logGP[node_id](0, next_parent_state, next_self_state);
    lp1 += logGP[node_id](1, next_parent_state, next_self_state);
  }

  // children and current level
  for (size_t count = 1; count < subtree_sizes[node_id];) {
    const size_t child_id = node_id + count;
    const size_t prev_child_state = meth_prev[child_id];
    const size_t curr_child_state = meth_curr[child_id];
    lp0 += logGP[child_id](prev_child_state, 0, curr_child_state);
    lp1 += logGP[child_id](prev_child_state, 1, curr_child_state);
    count += subtree_sizes[child_id];
  }

  return lp0 - lp1;
}


template <class T>
static double
MB_state_end(const std::vector<size_t> &subtree_sizes,
             const std::vector<size_t> &parent_ids,
             const pair_state &logG, // the matrix made of f0 & f1
             const std::vector<triple_state> &logGP,
             const std::vector<T> &meth_prev,
             const std::vector<T> &meth_curr, const size_t node_id) {

  double lp0 = 0.0, lp1 = 0.0;
  // parents in network
  if (is_root(node_id)) {
    const size_t prev_self_state = meth_prev[node_id];
    lp0 += logG(prev_self_state, 0);
    lp1 += logG(prev_self_state, 1);
  } else {
    const size_t curr_parent_state = meth_curr[parent_ids[node_id]];
    const size_t prev_self_state = meth_prev[node_id];
    lp0 += logGP[node_id](prev_self_state, curr_parent_state, 0);
    lp1 += logGP[node_id](prev_self_state, curr_parent_state, 1);
  }

  // children in network
  for (size_t count = 1; count < subtree_sizes[node_id];) {
    const size_t child_id = node_id + count;
    const T prev_child_state = meth_prev[child_id];
    const T curr_child_state = meth_curr[child_id];
    lp0 += logGP[child_id](prev_child_state, 0, curr_child_state);
    lp1 += logGP[child_id](prev_child_state, 1, curr_child_state);
    count += subtree_sizes[child_id];
  }

  return lp0 - lp1;
}


template <class T>
static double
MB_state_start(const std::vector<size_t> &subtree_sizes,
               const std::vector<size_t> &parent_ids,
               const std::pair<double, double> &log_pi,
               const pair_state &logG, // the matrix made of f0 & f1
               const std::vector<pair_state> &logP,
               const std::vector<triple_state> &logGP,
               const std::vector<T> &meth_curr,
               const std::vector<T> &meth_next, const size_t node_id) {

  double lp0 = 0.0, lp1 = 0.0;
  const size_t parent = parent_ids[node_id];
  // transition from parents in network [same here as parents in tree]
  // (or root distribution)
  if (is_root(node_id)) {
    lp0 += log_pi.first;
    lp1 += log_pi.second;
  } else {
    const T curr_parent_state = meth_curr[parent];
    lp0 += logP[node_id](curr_parent_state, 0);
    lp1 += logP[node_id](curr_parent_state, 1);
  }

  // children in network (only one parent; current self)
  for (size_t count = 1; count < subtree_sizes[node_id];) {
    const size_t child_id = node_id + count;
    const T curr_child_state = meth_curr[child_id];
    lp0 += logP[child_id](0, curr_child_state);
    lp1 += logP[child_id](1, curr_child_state);
    count += subtree_sizes[child_id];
  }

  // horizontal children in network and their other parents
  if (is_root(node_id)) {
    const T next_self_state = meth_next[node_id];
    lp0 += logG(0, next_self_state);
    lp1 += logG(1, next_self_state);
  } else {
    const T next_parent_state = meth_next[parent];
    const T next_self_state = meth_next[node_id];
    lp0 += logGP[node_id](0, next_parent_state, next_self_state);
    lp1 += logGP[node_id](1, next_parent_state, next_self_state);
  }
  return lp0 - lp1;
}


template <class T>
static double
MB_state_indep(const std::vector<size_t> &subtree_sizes,
               const std::vector<size_t> &parent_ids,
               const std::pair<double, double> &log_pi,
               const std::vector<pair_state> &logP,
               const std::vector<T> &meth_curr,
               const size_t node_id) {

  double lp0 = 0.0, lp1 = 0.0;
  const size_t parent = parent_ids[node_id];
  // transition from parents in network [same here as parents in tree]
  // (or root distribution)
  if (is_root(node_id)) {
    lp0 += log_pi.first;
    lp1 += log_pi.second;
  } else {
    const T curr_parent_state = meth_curr[parent];
    lp0 += logP[node_id](curr_parent_state, 0);
    lp1 += logP[node_id](curr_parent_state, 1);
  }

  // children in network (only one parent; current self)
  for (size_t count = 1; count < subtree_sizes[node_id];) {
    const size_t child_id = node_id + count;
    const T curr_child_state = meth_curr[child_id];
    lp0 += logP[child_id](0, curr_child_state);
    lp1 += logP[child_id](1, curr_child_state);
    count += subtree_sizes[child_id];
  }

  return lp0 - lp1;
}


template <class T>
void
epiphy_mcmc::accept_or_reject_proposal(const double log_ratio, const size_t idx,
                                       std::vector<T> &states) const {
  const T tmp_state = states[idx];
  const double ratio = (!tmp_state) ? 1.0/(exp(log_ratio) + 1.0) : 1.0/(exp(-log_ratio) + 1.0);
  if (dis(gen) < ratio)
    // ADS: need to make sure the "!" is defined the type T
    states[idx] = !tmp_state; // = (1 - tmp_state)
}


template <class T>
void
epiphy_mcmc::sample_states_middle(const std::vector<size_t> &subtree_sizes,
                                  const std::vector<size_t> &parent_ids,
                                  const pair_state &logG, // f0 & f1
                                  const std::vector<triple_state> &logGP,
                                  const std::vector<double> &marginals,
                                  const std::vector<T> &states_prev,
                                  std::vector<T> &states_curr,
                                  const std::vector<T> &states_next,
                                  const size_t node_id) const {

  if (!missing_meth_value(marginals[node_id])) { // site is observed
    states_curr[node_id] = dis(gen) > marginals[node_id] ? 0 : 1;
  } else { // internal node or missing data at leaf
    for (size_t count = 1; count < subtree_sizes[node_id];) {
      const size_t child_id = node_id + count;
      sample_states_middle(subtree_sizes, parent_ids, logG, logGP, marginals,
                           states_prev, states_curr, states_next, child_id);
      count += subtree_sizes[child_id];
    }
    const double log_ratio =
      MB_state_middle(subtree_sizes, parent_ids, logG, logGP,
                      states_prev, states_curr, states_next, node_id);

    accept_or_reject_proposal(log_ratio, node_id, states_curr);
  }
}


template <class T>
void
epiphy_mcmc::sample_states_end(const std::vector<size_t> &subtree_sizes,
                               const std::vector<size_t> &parent_ids,
                               const pair_state &logG, // the matrix made of f0 & f1
                               const std::vector<triple_state> &logGP,
                               const std::vector<double> &marginals,
                               const std::vector<T> &states_prev,
                               std::vector<T> &states_curr,
                               const size_t node_id) const {

  if (is_leaf(subtree_sizes[node_id]) &&
      !missing_meth_value(marginals[node_id])) {
    states_curr[node_id] = dis(gen) > marginals[node_id] ? 0 : 1;
  } else {
    for (size_t count = 1; count < subtree_sizes[node_id];) {
      const size_t child_id = node_id + count;
      sample_states_end(subtree_sizes, parent_ids, logG, logGP,
                        marginals, states_prev, states_curr, child_id);
      count += subtree_sizes[child_id];
    }
    const double log_ratio =
      MB_state_end(subtree_sizes, parent_ids, logG, logGP,
                   states_prev, states_curr, node_id);

    accept_or_reject_proposal(log_ratio, node_id, states_curr);
  }
}


template <class T>
void
epiphy_mcmc::sample_states_start(const std::vector<size_t> &subtree_sizes,
                                 const std::vector<size_t> &parent_ids,
                                 const std::pair<double, double> &log_pi,
                                 const pair_state &logG, // the matrix made of f0 & f1
                                 const std::vector<pair_state> &logP,
                                 const std::vector<triple_state> &logGP,
                                 const std::vector<double> &marginals,
                                 std::vector<T> &states_curr,
                                 const std::vector<T> &states_next,
                                 const size_t node_id) const {

  if (is_leaf(subtree_sizes[node_id]) &&
      !missing_meth_value(marginals[node_id])) {
    states_curr[node_id] = dis(gen) > marginals[node_id] ? 0 : 1;
  } else {
    for (size_t count = 1; count < subtree_sizes[node_id]; ) {
      const size_t child_id = node_id + count;
      sample_states_start(subtree_sizes, parent_ids, log_pi, logG, logP, logGP,
                          marginals, states_curr, states_next, child_id);
      count += subtree_sizes[child_id];
    }
    const double log_ratio =
      MB_state_start(subtree_sizes, parent_ids, log_pi, logG, logP, logGP,
                     states_curr, states_next, node_id);

    accept_or_reject_proposal(log_ratio, node_id, states_curr);
  }
}



////////////////////////////////////////////////////////////////////////////////
///////////////////////////// independent sampling /////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <class T>
void
epiphy_mcmc::sample_states_indep(const std::vector<size_t> &subtree_sizes,
                                 const std::vector<size_t> &parent_ids,
                                 const std::pair<double, double> &log_pi,
                                 const std::vector<pair_state> &logP,
                                 const std::vector<double> &marginals,
                                 std::vector<T> &states_curr,
                                 const size_t node_id) const {

  if (is_leaf(subtree_sizes[node_id]) &&
      !missing_meth_value(marginals[node_id])) {
    states_curr[node_id] = dis(gen) > marginals[node_id] ? 0 : 1;
  } else {
    for (size_t count = 1; count < subtree_sizes[node_id]; ) {
      const size_t child_id = node_id + count;
      sample_states_indep(subtree_sizes, parent_ids, log_pi, logP,
                          marginals, states_curr, child_id);
      count += subtree_sizes[child_id];
    }
    const double log_ratio =
      MB_state_indep(subtree_sizes, parent_ids, log_pi, logP,
                     states_curr, node_id);

    accept_or_reject_proposal(log_ratio, node_id, states_curr);
  }
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   sampling in blocks   ///////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <class T>
void
epiphy_mcmc::sample_states(const bool OBS,
                           const std::vector<size_t> &subtree_sizes,
                           const std::vector<size_t> &parent_ids,
                           const param_set &params,
                           const std::vector<std::vector<double> > &marginals,
                           const std::vector<std::pair<size_t, size_t> > &blocks,
                           std::vector<std::vector<T> > &states) const {

  std::pair<double, double> log_pi(std::make_pair(log(params.pi0),
                                                  log(1.0 - params.pi0)));
  pair_state logG(params.g0[0], 1.0 - params.g0[0], 1.0 - params.g1[0], params.g1[0]);
  logG.make_logs();

  std::vector<pair_state> logP;
  std::vector<triple_state> logGP;
  get_transition_matrices(params, logP, logGP);
  for (size_t i = 0; i < logP.size(); ++i) {
    logP[i].make_logs();
    logGP[i].make_logs();
  }

#pragma omp parallel for
  for (size_t i = 0; i < blocks.size(); ++i) {
    const size_t start = blocks[i].first;
    const size_t end = blocks[i].second;
    if (start < end) {
      // sample forward
      sample_states_start(subtree_sizes, parent_ids,
                          log_pi, logG, logP, logGP, marginals[start],
                          states[start], states[start + 1], 0);
      for (size_t pos = start + 1; pos < end; ++pos) {
        vector<T> state_prev = states[pos - 1];
        vector<T> state_next = states[pos + 1];
        if (OBS) {
          // change prev and next leaf states to
          // the closest observed sites' states
          for (size_t node_id = 0; node_id < subtree_sizes.size(); ++node_id) {
            if (is_leaf(subtree_sizes[node_id]) &&
                missing_meth_value(marginals[pos][node_id])) {
              size_t k;
              for (k = pos - 1;
                   k > start && missing_meth_value(marginals[k][node_id]); --k);
              if (k >= start && !missing_meth_value(marginals[k][node_id]))
                state_prev[node_id] = states[k][node_id];
              for (k = pos + 1;
                   k <= end && missing_meth_value(marginals[k][node_id]); ++k);
              if (k <= end && !missing_meth_value(marginals[k][node_id]))
                state_next[node_id] = states[k][node_id];
            }
          }
        }
        sample_states_middle(subtree_sizes, parent_ids,
                             logG, logGP, marginals[pos],
                             state_prev, states[pos], state_next, 0);
      }
      sample_states_end(subtree_sizes, parent_ids, logG, logGP, marginals[end],
                        states[end - 1], states[end], 0);
    } else {
      sample_states_indep(subtree_sizes, parent_ids, log_pi, logP,
                          marginals[start], states[start], 0);
    }
  }
}


template <class T>
void
epiphy_mcmc::sample_states_rev(const bool OBS,
                               const std::vector<size_t> &subtree_sizes,
                               const std::vector<size_t> &parent_ids,
                               const param_set &params,
                               const std::vector<std::vector<double> > &marginals,
                               const std::vector<std::pair<size_t, size_t> > &blocks,
                               std::vector<std::vector<T> > &states) const {

  std::pair<double, double> log_pi(std::make_pair(log(params.pi0),
                                                  log(1.0 - params.pi0)));
  pair_state logG(params.g0[0], 1.0 - params.g0[0], 1.0 - params.g1[0], params.g1[0]);
  logG.make_logs();

  std::vector<pair_state> logP;
  std::vector<triple_state> logGP;
  get_transition_matrices(params, logP, logGP);
  for (size_t i = 0; i < logP.size(); ++i) {
    logP[i].make_logs();
    logGP[i].make_logs();
  }

#pragma omp parallel for
  for (size_t i = 0; i < blocks.size(); ++i) {
    const size_t start = blocks[i].first;
    const size_t end = blocks[i].second;
    if (start < end) {
      // sample backward
      sample_states_start(subtree_sizes, parent_ids, log_pi, logG, logP, logGP,
                          marginals[end], states[end], states[end - 1], 0);
      for (size_t j = 1; j < end - start; ++j) {
        const size_t pos = end - j;
        vector<T> state_prev = states[pos - 1];
        vector<T> state_next = states[pos + 1];
        if (OBS) {
          // change prev and next leaf states to
          // the closest observed sites' states
          for (size_t node_id = 0; node_id < subtree_sizes.size(); ++node_id) {
            if (is_leaf(subtree_sizes[node_id]) &&
                missing_meth_value(marginals[pos][node_id])) {
              size_t k = pos - 1;
              for (k = pos -1;
                   k > start && missing_meth_value(marginals[k][node_id]); --k);

              if (k >= start && !missing_meth_value(marginals[k][node_id]))
                state_prev[node_id] = states[k][node_id];

              for (k = pos + 1;
                   k <= end && missing_meth_value(marginals[k][node_id]); ++k);

              if (k <= end && !missing_meth_value(marginals[k][node_id]))
                state_next[node_id] = states[k][node_id];
            }
          }
        }
        sample_states_middle(subtree_sizes, parent_ids, logG, logGP,
                             marginals[pos], state_next, states[pos],
                             state_prev, 0);
      }
      sample_states_end(subtree_sizes, parent_ids, logG, logGP, marginals[start],
                        states[start + 1], states[start], 0);
    } else {
      sample_states_indep(subtree_sizes, parent_ids, log_pi, logP,
                          marginals[start], states[start], 0);
    }
  }
}


// construct transition probabilities directly from sufficient statistics
template <class T>
void
epiphy_mcmc::sample_states(const bool OBS,
                           const std::vector<size_t> &subtree_sizes,
                           const std::vector<size_t> &parent_ids,
                           const suff_stat &ss,
                           const std::vector<std::vector<double> > &marginals,
                           const std::vector<std::pair<size_t, size_t> > &blocks,
                           std::vector<std::vector<T> > &states) const {
  const double denom = ss.monad_root.first + ss.monad_root.second;
  std::pair<double, double> log_pi =
    std::make_pair(log(ss.monad_root.first/denom),
                   log(ss.monad_root.second/denom));
  pair_state logG(ss.dyad_root);
  logG.to_probabilities();
  logG.make_logs();

  size_t n_nodes = subtree_sizes.size();
  std::vector<pair_state> logP;
  std::vector<triple_state> logGP;
  for (size_t i = 0; i < n_nodes; ++i) {
    logP.push_back(ss.dyads[i]);
    logP[i].to_probabilities();
    logP[i].make_logs();
    logGP.push_back(ss.triads[i]);
    logGP[i].to_probabilities();
    logGP[i].make_logs();
  }

#pragma omp parallel for
  for (size_t i = 0; i < blocks.size(); ++i) {
    const size_t start = blocks[i].first;
    const size_t end = blocks[i].second;
    if (start < end) {
      sample_states_start(subtree_sizes, parent_ids, log_pi, logG, logP, logGP,
                          marginals[start], states[start], states[start + 1], 0);
      for (size_t pos = start + 1; pos < end; ++pos) {
        vector<T> state_prev = states[pos - 1];
        vector<T> state_next = states[pos + 1];
        if (OBS) {
          // change prev and next leaf states to
          // the closest observed sites' states
          for (size_t node_id = 0; node_id < subtree_sizes.size(); ++node_id) {
            if (is_leaf(subtree_sizes[node_id]) &&
                missing_meth_value(marginals[pos][node_id])) {
              size_t k;
              for (k = pos - 1;
                   k > start && missing_meth_value(marginals[k][node_id]); --k);
              if (k >= start && !missing_meth_value(marginals[k][node_id]))
                state_prev[node_id] = states[k][node_id];
              for (k = pos + 1;
                   k <= end && missing_meth_value(marginals[k][node_id]); ++k);
              if (k <= end && !missing_meth_value(marginals[k][node_id]))
                state_next[node_id] = states[k][node_id];
            }
          }
        }
        sample_states_middle(subtree_sizes, parent_ids, logG, logGP,
                             marginals[pos], state_prev, states[pos],
                             state_next, 0);
      }
      sample_states_end(subtree_sizes, parent_ids, logG, logGP, marginals[end],
                        states[end - 1], states[end], 0);
    } else {
      sample_states_indep(subtree_sizes, parent_ids, log_pi, logP,
                          marginals[end], states[start], 0);
    }
  }
}


// this is redundant to count_triads
template <class T>
static void
collect_sample_stat(const std::vector<size_t> &subtree_sizes,
                    const std::vector<size_t> &parent_ids,
                    const std::vector<std::vector<T> > &tree_states,
                    const std::vector<std::pair<size_t, size_t> > &reset_points,
                    suff_stat &sample_stat) {
  get_suff_stat(subtree_sizes, parent_ids, tree_states,
                reset_points, sample_stat);
}


void
sum(const std::vector<suff_stat> &mcmcstats, suff_stat &sum_ss);


void
average(const std::vector<suff_stat> &mcmcstats, suff_stat &ave_ss);


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////// MCMC output analysis    /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// measure MCMC convergence
bool
CBM_convergence(const std::vector<suff_stat> &mcmcstats, double &test_val,
                size_t &batch_size, size_t &batch_number);

#endif
