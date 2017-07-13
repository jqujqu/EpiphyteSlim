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

#include "epiphy_mcmc.hpp"
#include "epiphy_utils.hpp"
#include "sufficient_statistics_helpers.hpp"  // suff_stat

#include <vector>
#include <random>
#include <iostream>
#include <math.h>       /* pow */
#include <algorithm>  //std::max, min
#include <boost/math/distributions/students_t.hpp>

using std::vector;
using std::cerr;
using std::endl;
using std::max;

#include <functional>
using std::placeholders::_1;
using std::bind;
using std::plus;

// static const double PROBABILITY_GUARD = 1e-10;
static const double CBM_THETA = 0.5;
static const double CBM_EPS = 1e-4;


void
sum(const vector<suff_stat> &mcmcstats,
    suff_stat &sum_ss) {
  const size_t n_nodes = mcmcstats[0].dyads.size();
  const size_t N = mcmcstats.size();
  sum_ss = mcmcstats[0];
  for (size_t i = 1; i < N; ++i) {
    sum_ss.monad_root.first += mcmcstats[i].monad_root.first;
    sum_ss.monad_root.second += mcmcstats[i].monad_root.second;
    sum_ss.dyad_root += mcmcstats[i].dyad_root;
    for (size_t j = 0; j < n_nodes; ++j) {
      sum_ss.dyads[j] += mcmcstats[i].dyads[j];
      sum_ss.triads[j] += mcmcstats[i].triads[j];
    }
  }
}


void
average(const vector<suff_stat> &mcmcstats, suff_stat &ave_ss) {
  const size_t n_nodes = mcmcstats[0].dyads.size();
  sum(mcmcstats, ave_ss);
  const size_t N = mcmcstats.size();
  ave_ss.monad_root.first /= N;
  ave_ss.monad_root.second /= N;
  ave_ss.dyad_root.div(N);
  for (size_t j = 0; j < n_nodes; ++j) {
    ave_ss.dyads[j].div(N);
    ave_ss.triads[j].div(N);
  }
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//////////          MCMC output analysis             ///////////////////////////
///////// single chain, Batch Mean with increasing batch size //////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void
MCMC_MSE(const vector<suff_stat> &mcmcstats,
         const double CBM_THETA,
         const double CBM_EPS,
         double &test_val, size_t &b, size_t &a,
         bool &stop) {

  const size_t n = mcmcstats.size();
  const size_t n_sites = (mcmcstats[0].dyad_root.uu + mcmcstats[0].dyad_root.um +
                          mcmcstats[0].dyad_root.mu + mcmcstats[0].dyad_root.mm);

  /*determine batch number a and batch size b*/
  b = static_cast<size_t>(floor(pow(n, CBM_THETA)));
  a = static_cast<size_t>(floor(double(n)/b));

  /*whole chain average*/
  suff_stat ave_mcmcstats;
  vector<suff_stat> mcmcstats_ab(mcmcstats.end()-a*b, mcmcstats.end());
  average(mcmcstats_ab, ave_mcmcstats);

  /*batch means*/
  vector<suff_stat> mcmcstats_batch_means;
  for (size_t j = 0; j < a; ++j) {
    vector<suff_stat> batch(mcmcstats.end() - (j+1)*b, mcmcstats.end() - j*b);
    suff_stat batch_mean;
    average(batch, batch_mean);
    mcmcstats_batch_means.push_back(batch_mean);
  }

  /*mean squared errors*/
  suff_stat mse = ave_mcmcstats; // (re)initialize
  const size_t n_nodes = mcmcstats[0].dyads.size();
  // only care about root_counts  start_counts and triad_counts
  for (size_t j = 0; j < a; ++j) {
    mse.dyad_root = ((mcmcstats_batch_means[j].dyad_root -
                      ave_mcmcstats.dyad_root) *
                     (mcmcstats_batch_means[j].dyad_root -
                      ave_mcmcstats.dyad_root));
    mse.dyad_root.div(double(a-1)/b);
    for (size_t node_id = 0; node_id < n_nodes; ++node_id) {
      pair_state start_counts = ((mcmcstats_batch_means[j].dyads[node_id] -
                                  ave_mcmcstats.dyads[node_id]) *
                                 (mcmcstats_batch_means[j].dyads[node_id] -
                                  ave_mcmcstats.dyads[node_id]));
      start_counts.div(double(a-1)/b);
      mse.dyads[node_id] += start_counts;
      triple_state triad_counts = ((mcmcstats_batch_means[j].triads[node_id] -
                                    ave_mcmcstats.triads[node_id]) *
                                   (mcmcstats_batch_means[j].triads[node_id] -
                                    ave_mcmcstats.triads[node_id]));
      triad_counts.div(double(a-1)/b);
      mse.triads[node_id] += triad_counts;
    }
  }

  boost::math::students_t dist(a - 1);
  double T = boost::math::quantile(complement(dist, 0.05/2));

  double cbm_max_mse = 0;
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 2; ++j)
      cbm_max_mse = max(cbm_max_mse, mse.dyad_root(i, j));
  for (size_t node_id = 0; node_id < n_nodes; ++node_id) {
    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 2; ++j) {
        cbm_max_mse = max(cbm_max_mse, mse.dyad_root(i, j));
        cbm_max_mse = max(cbm_max_mse, mse.dyads[node_id](i, j));
        for (size_t k = 0; k < 2; ++k) {
          cbm_max_mse = max(cbm_max_mse, mse.triads[node_id](i, j, k));
        }
      }
    }
  }
  test_val = pow(cbm_max_mse/(a*b), 0.5)*T;
  stop = (test_val/n_sites < CBM_EPS);
}


bool
CBM_convergence(const vector<suff_stat> &mcmcstats,
                double &test_val, size_t &batch_size, size_t &batch_number) {
  test_val = 0;
  batch_size = 0;
  batch_number = 0;
  bool converged;
  MCMC_MSE(mcmcstats, CBM_THETA, CBM_EPS,
           test_val, batch_size, batch_number, converged);

  return converged;
}

