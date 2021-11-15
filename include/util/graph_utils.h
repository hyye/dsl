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

#ifndef DSL_GRAPH_UTILS_H
#define DSL_GRAPH_UTILS_H

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/graph_utility.hpp>

namespace dsl {

class GraphUtils {
  typedef boost::adjacency_list_traits<boost::vecS, boost::vecS,
                                       boost::directedS>
      Traits;
  typedef boost::adjacency_list<
      boost::vecS, boost::vecS, boost::directedS,
      boost::property<
          boost::vertex_name_t, std::string,
          boost::property<
              boost::vertex_index_t, long,
              boost::property<
                  boost::vertex_color_t, boost::default_color_type,
                  boost::property<
                      boost::vertex_distance_t, long,
                      boost::property<boost::vertex_predecessor_t,
                                      Traits::edge_descriptor> > > > >,

      boost::property<
          boost::edge_capacity_t, long,
          boost::property<boost::edge_residual_capacity_t, long,
                          boost::property<boost::edge_reverse_t,
                                          Traits::edge_descriptor> > > >
      Graph;

  typedef
      typename boost::graph_traits<Graph>::vertex_descriptor vertex_descriptor;
  typedef typename boost::graph_traits<Graph>::edge_descriptor edge_descriptor;
  using VertexColors = boost::property_map<Graph, boost::vertex_color_t>::type;

 public:
  void SetGraph(const std::map<std::pair<long, long>, long> &edge_values, long num_vertex);
  void CalcSourceSet(long source_idx, long target_idx, std::vector<long> &source_set);
  bool IsInSourceSet(vertex_descriptor v) const;
  Graph g;
  std::vector<vertex_descriptor> verts;

  VertexColors colors;
  boost::default_color_type src_color;
};

}  // namespace dsl

#endif  // DSL_GRAPH_UTILS_H
