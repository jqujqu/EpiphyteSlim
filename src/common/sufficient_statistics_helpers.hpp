/*    Copyright (C) 2016 University of Southern California and
 *                       Andrew D. Smith and Jenny Qu
 *
 *    Authors: Jenny Qu and Andrew Smith
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef SUFFICIENT_STATISTICS_HELPERS_HPP
#define SUFFICIENT_STATISTICS_HELPERS_HPP

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////  STRUCTS FOR HOLDING PAIRS AND TRIPLES OF STATES  ////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#include "MethpipeSite.hpp"

#include <vector>
#include <string>
#include <cmath>
#include <sstream>

struct param_set;

struct pair_state {
  double uu, um; // think 2x2 matrix
  double mu, mm;

  pair_state(double _uu, double _um, double _mu, double _mm) :
    uu(_uu), um(_um), mu(_mu), mm(_mm) {}
  pair_state() : uu(0.0), um(0.0), mu(0.0), mm(0.0) {}

  // WARNING: no range checking (for i,j > 1)
  double operator()(int i, int j) const { // version for l-values
    return (i == 0) ? (j == 0 ? uu : um) : (j == 0 ? mu : mm);
  }
  double & operator()(int i, int j) { // version for r-values
    return (i == 0) ? (j == 0 ? uu : um) : (j == 0 ? mu : mm);
  }
  void operator+=(const pair_state &other) {
    uu += other.uu; um += other.um;
    mu += other.mu; mm += other.mm;
  }
  void operator/=(const pair_state &other) {
    uu /= other.uu; um /= other.um;
    mu /= other.mu; mm /= other.mm;
  }
  void div(const double x) {
    uu /= x; um /= x;
    mu /= x; mm /= x;
  }
  void to_probabilities() {
    const double u_denom = uu + um;
    uu /= u_denom;
    um /= u_denom;
    const double m_denom = mu + mm;
    mu /= m_denom;
    mm /= m_denom;
  }
  void make_logs() {
    uu = std::log(uu); um = std::log(um);
    mu = std::log(mu); mm = std::log(mm);
  }
  void flatten(std::vector<double> &p) const {
    p.clear();
    p.push_back(uu); p.push_back(um);
    p.push_back(mu); p.push_back(mm);
  }
  std::string tostring() const {
    std::ostringstream oss;
    oss << "[" << uu << ", " << um << "]\n"
        << "[" << mu << ", " << mm << "]";
    return oss.str();
  }
};

std::ostream &
operator<<(std::ostream &out, const pair_state &ps);

struct triple_state {
  double uuu, uum; // think: a 2x2 matrix
  double umu, umm;

  double muu, mum; // think: another 2x2 matrix
  double mmu, mmm;

  triple_state() : uuu(0.0), uum(0.0), umu(0.0), umm(0.0),
                   muu(0.0), mum(0.0), mmu(0.0), mmm(0.0) {};
  triple_state(double _uuu, double _uum, double _umu, double _umm,
               double _muu, double _mum, double _mmu, double _mmm) :
    uuu(_uuu), uum(_uum), umu(_umu), umm(_umm),
    muu(_muu), mum(_mum), mmu(_mmu), mmm(_mmm) {};

  // WARNING: no range checking
  double operator()(int i, int j, int k) const { // version for r-value (rhs)
    return  i == 0 ?
      (j == 0 ? (k == 0 ? uuu : uum) : (k == 0 ? umu : umm)) :
      (j == 0 ? (k == 0 ? muu : mum) : (k == 0 ? mmu : mmm));
  }
  double & operator()(int i, int j, int k) { // version for l-value (lhs)
    return  i == 0 ?
      (j == 0 ? (k == 0 ? uuu : uum) : (k == 0 ? umu : umm)) :
      (j == 0 ? (k == 0 ? muu : mum) : (k == 0 ? mmu : mmm));
  }
  triple_state operator+(const triple_state &other) const {
    return triple_state(uuu + other.uuu, uum + other.uum,
                        umu + other.umu, umm + other.umm,
                        muu + other.muu, mum + other.mum,
                        mmu + other.mmu, mmm + other.mmm);
  }
  triple_state operator-(const triple_state &other) const {
    return triple_state(uuu - other.uuu, uum - other.uum,
                        umu - other.umu, umm - other.umm,
                        muu - other.muu, mum - other.mum,
                        mmu - other.mmu, mmm - other.mmm);
  }
  triple_state operator*(const triple_state &other) const {
    return triple_state(uuu*other.uuu, uum*other.uum,
                        umu*other.umu, umm*other.umm,
                        muu*other.muu, mum*other.mum,
                        mmu*other.mmu, mmm*other.mmm);
  }
  void operator+=(const triple_state &other) {
    uuu += other.uuu; uum += other.uum;
    umu += other.umu; umm += other.umm;
    muu += other.muu; mum += other.mum;
    mmu += other.mmu; mmm += other.mmm;
  }
  void operator/=(const triple_state &other) {
    uuu /= other.uuu; uum /= other.uum;
    umu /= other.umu; umm /= other.umm;
    muu /= other.muu; mum /= other.mum;
    mmu /= other.mmu; mmm /= other.mmm;
  }
  void to_probabilities() {
    const double uu_denom = uuu + uum;
    uuu /= uu_denom;
    uum /= uu_denom;
    const double um_denom = umu + umm;
    umu /= um_denom;
    umm /= um_denom;
    const double mu_denom = muu + mum;
    muu /= mu_denom;
    mum /= mu_denom;
    const double mm_denom = mmu + mmm;
    mmu /= mm_denom;
    mmm /= mm_denom;
  }
  void div(const double x) {
    uuu /= x; uum /= x;
    umu /= x; umm /= x;

    muu /= x; mum /= x;
    mmu /= x; mmm /= x;
  }
  void make_logs() {
    uuu = std::log(uuu); uum = std::log(uum);
    umu = std::log(umu); umm = std::log(umm);

    muu = std::log(muu); mum = std::log(mum);
    mmu = std::log(mmu); mmm = std::log(mmm);
  }
  void flatten(std::vector<double> &p) const {
    p.clear();
    p.push_back(uuu); p.push_back(uum);
    p.push_back(umu); p.push_back(umm);
    p.push_back(muu); p.push_back(mum);
    p.push_back(mmu); p.push_back(mmm);
  }
  std::string tostring() const {
    std::ostringstream oss;
    oss << "u [" << uuu << ", " << uum << "]\n"
        << "  [" << umu << ", " << umm << "]\n"
        << "m [" << muu << ", " << mum << "]\n"
        << "  [" << mmu << ", " << mmm << "]";
    return oss.str();
  }
};

std::ostream &
operator<<(std::ostream &out, const triple_state &ts);


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////  STRUCT FOR HOLDING SUFFICIENT STATISTICS   ///////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
struct suff_stat {
  std::pair<double, double> monad_root;
  pair_state dyad_root;
  std::vector<pair_state> dyads;
  std::vector<triple_state> triads;

  suff_stat() {}
  
  suff_stat(const size_t n_nodes) {
    monad_root = std::make_pair(0.0, 0.0);
    dyad_root = pair_state();
    dyads = std::vector<pair_state>(n_nodes);
    triads = std::vector<triple_state>(n_nodes);
  }

  suff_stat(const std::pair<double, double> &mr,
            const pair_state &dr,
            const std::vector<pair_state> &ds,
            const std::vector<triple_state> &ts) :
    monad_root(mr), dyad_root(dr), dyads(ds), triads(ts) {}

  std::string tostring() const {
    std::ostringstream oss;
    oss << "monad_root\t["
        << monad_root.first << ", "
        << monad_root.second << "]\n";
    oss << "dyad_root\n"
        << dyad_root.tostring() << "\n";
    oss << "dyads\n";
    for (size_t i = 1; i < dyads.size(); ++i) {
      oss << dyads[i].tostring() << "\tnode_id=" << i << "\n";
    }
    oss << "triads\n";
    for (size_t i = 1; i < triads.size(); ++i) {
      oss << triads[i].tostring() << "\tnode_id=" << i << "\n";
    }
    return oss.str();
  }
};

////////////////////////////////////////////////////////////////////////////////

void
get_transition_matrices(const param_set &ps,
                        std::vector<pair_state> &P,
                        std::vector<triple_state> &GP);

void
get_transition_matrices_deriv(const param_set &ps,
                              std::vector<pair_state> &P,
                              std::vector<triple_state> &GP,
                              std::vector<triple_state> &GP_drate,
                              std::vector<triple_state> &GP_dg0,
                              std::vector<triple_state> &GP_dg1,
                              std::vector<triple_state> &GP_dT);

/* The count_triads function resides in the header and not cpp file
   because it is templated. */
template <class T>
static void
get_suff_stat(const std::vector<size_t> &subtree_sizes,
              const std::vector<size_t> &parent_ids,
              const std::vector<std::vector<T> > &tree_states,
              const std::vector<std::pair<size_t, size_t> > &reset_points,
              suff_stat &ss) {
  for (size_t i = 0; i < reset_points.size(); ++i) {
    const size_t start = reset_points[i].first;
    const size_t end = reset_points[i].second;
    ss.monad_root.first += !tree_states[start][0];
    ss.monad_root.second += tree_states[start][0];
    for (size_t node_id = 1; node_id < subtree_sizes.size(); ++node_id) {
      const size_t parent = tree_states[start][parent_ids[node_id]];
      const size_t curr = tree_states[start][node_id];
      ss.dyads[node_id](parent, curr) += 1.0;
    }
    for (size_t pos = start + 1; pos <= end; ++pos) {
      ss.dyad_root(tree_states[pos - 1][0], tree_states[pos][0])++;
      for (size_t node_id = 1; node_id < subtree_sizes.size(); ++node_id) {
        const size_t parent = tree_states[pos][parent_ids[node_id]];
        const size_t prev = tree_states[pos - 1][node_id];
        const size_t curr = tree_states[pos][node_id];
        ss.triads[node_id](prev, parent, curr) += 1.0;
      }
    }
  }
}


template <class T>
static void
get_dinuc_stat(const std::vector<size_t> &subtree_sizes,
               const std::vector<size_t> &parent_ids,
               const std::vector<std::vector<T> > &tree_states,
               std::vector<std::vector<double> > &root_dinuc,
               std::vector<std::vector<std::vector<double> > > &pair_dinuc) {
  const size_t n_nodes = subtree_sizes.size();
  root_dinuc = std::vector<std::vector<double> >(2, std::vector<double>(2, 0.0));
  std::vector<std::vector<double> > pdtmp(4, std::vector<double>(4, 0.0));
  pair_dinuc = std::vector<std::vector<std::vector<double> > >(n_nodes, pdtmp);
  for (size_t pos = 0 ; pos < tree_states.size() - 1; ++pos) {
    root_dinuc[tree_states[pos][0]][tree_states[pos+1][0]]++;
    for (size_t node_id = 1; node_id < subtree_sizes.size(); ++node_id) {
      const size_t parent_curr = tree_states[pos][parent_ids[node_id]];
      const size_t parent_next = tree_states[pos+1][parent_ids[node_id]];
      const size_t curr = tree_states[pos][node_id];
      const size_t next = tree_states[pos+1][node_id];
      pair_dinuc[node_id][2*parent_curr + parent_next][2*curr + next] += 1.0;
    }
  }
}


template <class T>
static void
get_dinuc_stat(const std::vector<size_t> &subtree_sizes,
               const std::vector<size_t> &parent_ids,
               const std::vector<std::vector<T> > &tree_states,
               const std::vector<std::pair<size_t, size_t> > &reset_points,
               std::vector<std::vector<double> > &root_dinuc,
               std::vector<std::vector<std::vector<double> > > &pair_dinuc) {
  const size_t n_nodes = subtree_sizes.size();
  root_dinuc = std::vector<std::vector<double> >(2, std::vector<double>(2, 0.0));
  std::vector<std::vector<double> > pdtmp(4, std::vector<double>(4, 0.0));
  pair_dinuc = std::vector<std::vector<std::vector<double> > >(n_nodes, pdtmp);
  for (size_t i = 0; i < reset_points.size(); ++i) {
    const size_t start = reset_points[i].first;
    const size_t end = reset_points[i].second;
    for (size_t pos = start ; pos < end - 1; ++pos) {
      root_dinuc[tree_states[pos][0]][tree_states[pos+1][0]]++;
      for (size_t node_id = 1; node_id < subtree_sizes.size(); ++node_id) {
        const size_t parent_curr = tree_states[pos][parent_ids[node_id]];
        const size_t parent_next = tree_states[pos+1][parent_ids[node_id]];
        const size_t curr = tree_states[pos][node_id];
        const size_t next = tree_states[pos+1][node_id];
        pair_dinuc[node_id][2*parent_curr + parent_next][2*curr + next] += 1.0;
      }
    }
  }
}



std::string
dinuc_stat_tostring(const std::vector<std::vector<double> > &root_dinuc,
                    const std::vector<std::vector<std::vector<double> > > &pair_dinuc);
#endif
