/*****************************************************************************
 *  Copyright (C) 2016 University of Southern California and
 *                     Jenny Qu and Andrew D Smith
 *
 *  Authors: Jenny Qu and Andrew D Smith
 *
 *  This program is free software: you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License as
 *  published by the Free Software Foundation, either version 3 of the
 *  License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 *  General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see
 *  <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#ifndef OPTIMIZE_PARAMS_HPP
#define OPTIMIZE_PARAMS_HPP

class param_set;
class triple_state;
class pair_state;
struct suff_stat;

double
log_likelihood(const std::vector<size_t> &subtree_sizes, const param_set &ps,
               const suff_stat &ss); 

void
optimize_params(const bool VERBOSE, const size_t HORIZ_MODE,
                const std::vector<size_t> &subtree_sizes, const suff_stat &ss,
                param_set &ps);

void
max_likelihood_pi0(const bool VERBOSE, const suff_stat &ss, param_set &ps);

void
max_likelihood_horiz(const bool VERBOSE, const size_t HORIZ_MODE, 
                     const std::vector<size_t> &subtree_sizes,
                     const suff_stat &ss, param_set &ps);

void
max_likelihood_rate(const bool VERBOSE,
                    const std::vector<size_t> &subtree_sizes,
                    const suff_stat &ss, param_set &ps);

void
max_likelihood_branch(const bool VERBOSE,
                      const std::vector<size_t> &subtree_sizes,
                      const size_t node_id,  const suff_stat &ss,
                      param_set &ps);

void
max_likelihood_params(const bool VERBOSE,
                      const size_t HORIZ_MODE,                      
                      const std::vector<size_t> &subtree_sizes,
                      const suff_stat &ss, param_set &ps);

#endif
