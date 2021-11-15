/**
* This file is part of Direct Sparse Localization (DSL).
*
* Copyright (C) 2021 Haoyang Ye <hy.ye at connect dot ust dot hk>,
* and Huaiyang Huang <hhuangat at connect dot use dot hk>,
* Robotics and Multiperception Lab (RAM-LAB <https://ram-lab.com>),
* The Hong Kong University of Science and Technology
*
* For more information please see <https://github.com/hyye/dsl>.
* If you use this code, please cite the respective publications as
* listed on the above websites.
*
* DSL is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSL is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSL.  If not, see <http://www.gnu.org/licenses/>.
*/
//
// Created by hyye on 7/9/20.
//

#include "util/graph_utils.h"
#include "util/util_common.h"

using namespace boost;
namespace dsl {

bool GraphUtils::IsInSourceSet(vertex_descriptor v) const {
  return (src_color == get(colors, v));
}

void GraphUtils::SetGraph(
    const std::map<std::pair<long, long>, long> &edge_values, long num_vertex) {
  property_map<Graph, edge_capacity_t>::type capacity = get(edge_capacity, g);
  property_map<Graph, edge_residual_capacity_t>::type residual_capacity =
      get(edge_residual_capacity, g);
  property_map<Graph, edge_reverse_t>::type rev = get(edge_reverse, g);

  LOG(INFO) << "num_vertex: " << num_vertex;
  LOG(INFO) << "edge_values: " << edge_values.size();

  for (long vi = 0; vi < num_vertex; ++vi) {
    verts.push_back(add_vertex(g));
  }

  for (auto &&edge : edge_values) {
    long t0 = edge.first.first;
    long t1 = edge.first.second;
    edge_descriptor e1, e2;
    bool in1, in2;
    boost::tie(e1, in1) = add_edge(verts[t0], verts[t1], g);  // tail -> head
    boost::tie(e2, in2) = add_edge(verts[t1], verts[t0], g);

    //    if (!in1 || !in2) {
    //      LOG(ERROR) << "unable to add edge (" << t1 << "," << t0 << ")"
    //                 << std::endl;
    //      return;
    //    }
    capacity[e1] += edge.second;
    capacity[e2] += 0;
    rev[e1] = e2;
    rev[e2] = e1;
    //    LOG(INFO) << t0 << " " << t1 << " " << edge.second;
  }
  std::vector<default_color_type> color(num_vertices(g));
  std::vector<long> distance(num_vertices(g));
}

void GraphUtils::CalcSourceSet(long source_idx, long target_idx,
                               std::vector<long> &source_set) {
  source_set.clear();
  Traits::vertex_descriptor s, t;
  s = verts[source_idx];
  t = verts[target_idx];
  long flow = boykov_kolmogorov_max_flow(g, s, t);

  LOG(INFO) << "c  The total flow:";
  LOG(INFO) << "s " << flow;
  LOG(INFO) << "num_vertices(g) " << num_vertices(g);

  colors = get(boost::vertex_color, g);
  src_color = get(colors, s);
  graph_traits<Graph>::vertex_iterator u_iter, u_end;
  graph_traits<Graph>::out_edge_iterator ei, e_end;
  for (boost::tie(u_iter, u_end) = vertices(g); u_iter != u_end; ++u_iter) {
    if (IsInSourceSet(*u_iter)) {
      source_set.push_back(*u_iter);
    }
  }
}

}  // namespace dsl