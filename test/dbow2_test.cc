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
// Created by hyye on 6/30/20.
//

// Adapted from DBoW2
/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include "dsl_common.h"
#include "DBoW2/DBoW2.h"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "relocalization/vocabulary_binary.h"
#include "relocalization/feature_extractor.h"

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace DBoW2;
using namespace std;
using namespace dsl;
using namespace dsl::relocalization;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void loadFeatures(vector<vector<cv::Mat> > &features);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testVocCreation(const vector<vector<cv::Mat> > &features);
void testDatabase(const vector<vector<cv::Mat> > &features);
template<class TDescriptor, class F>
void testDatabaseVoc(const TemplatedVocabulary<TDescriptor, F> &voc, const vector<vector<cv::Mat> > &features);

void loadOrbFeatures(FeatureExtractor &extractor, vector<vector<cv::Mat> > &features);

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// number of training images
const int NIMAGES = 4;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void wait() {
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

// TEST(DBoW2Test, DemoTest) {
//   vector<vector<cv::Mat> > features;
//   loadFeatures(features);
//
//   testVocCreation(features);
//
//   // wait();
//
//   testDatabase(features);
// }

// TEST(DBoW2Test, LoadTxtBoostTest) {
//   VocabularyBinaryBoost<DBoW2::FORB::TDescriptor, DBoW2::FORB> voc;
//   voc.loadFromTextFile("../../support_files/vocabulary/ORBvoc.txt");
//   voc.save("../../support_files/vocabulary/ORBvoc.yml.gz");
//   voc.saveToBinaryFile("../../support_files/vocabulary/ORBvoc.boost.bin");
// }
//
// TEST(DBoW2Test, LoadYmlGzTest) {
//   TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> voc("../../support_files/vocabulary/ORBvoc.yml.gz");
//   LOG(INFO) << "Loaded!";
// }

// TEST(DBoW2Test, LoadBinBoostTest) {
//   VocabularyBinaryBoost<DBoW2::FORB::TDescriptor, DBoW2::FORB> voc;
//   voc.loadFromBinaryFile("../../support_files/vocabulary/ORBvoc.boost.bin");
//   LOG(INFO) << "Binary loaded!";
// }

// TEST(DBoW2Test, VocBoostTest) {
//   VocabularyBinaryBoost<DBoW2::FORB::TDescriptor, DBoW2::FORB> voc_bin;
//   voc_bin.loadFromBinaryFile("../../support_files/vocabulary/ORBvoc.boost.bin");
//   LOG(INFO) << voc_bin.GetK() << " " << voc_bin.GetL() << " " << voc_bin.GetWeighting() << " " << voc_bin.GetScoring()
//             << " " << voc_bin.GetNodes().size();
//
//   VocabularyBinaryBoost<DBoW2::FORB::TDescriptor, DBoW2::FORB> voc_txt;
//   voc_txt.loadFromTextFile("../../support_files/vocabulary/ORBvoc.txt");
//   EXPECT_EQ(voc_bin.GetK(), voc_txt.GetK());
//   EXPECT_EQ(voc_bin.GetL(), voc_txt.GetL());
//   EXPECT_EQ(voc_bin.GetScoring(), voc_txt.GetScoring());
//   EXPECT_EQ(voc_bin.GetWeighting(), voc_txt.GetWeighting());
//   EXPECT_EQ(voc_bin.GetNodes().size(), voc_txt.GetNodes().size());
//   EXPECT_EQ(cv::norm(voc_bin.GetNodes()[100].descriptor - voc_txt.GetNodes()[100].descriptor), 0);
//
//   VocabularyBinaryBoost<DBoW2::FORB::TDescriptor, DBoW2::FORB>::NodeSerializable n_bin = voc_bin.GetNodes()[100];
//   LOG(INFO) << n_bin.descriptor;
//   LOG(INFO) << voc_txt.GetNodes()[100].descriptor;
// }

// TEST(DBoW2Test, LoadTxtBoostTest) {
//   VocabularyBinaryBoost<DBoW2::FORB::TDescriptor, DBoW2::FORB> voc;
//   voc.loadFromTextFile("../../support_files/vocabulary/ORBvoc.txt");
//   voc.save("../../support_files/vocabulary/ORBvoc.yml.gz");
//   voc.saveToBinaryFile("../../support_files/vocabulary/ORBvoc.boost.bin");
// }

// TEST(DBoW2Test, DemoBinBoostTest) {
//   vector<vector<cv::Mat> > features;
//   loadFeatures(features);
//
//   VocabularyBinaryBoost<DBoW2::FORB::TDescriptor, DBoW2::FORB> voc_bin;
//   voc_bin.loadFromBinaryFile("../../support_files/vocabulary/ORBvoc.boost.bin");
//   LOG(INFO) << voc_bin.GetK() << " " << voc_bin.GetL() << " " << voc_bin.GetWeighting() << " " << voc_bin.GetScoring()
//             << " " << voc_bin.GetNodes().size();
//
//   VocabularyBinaryBoost<DBoW2::FORB::TDescriptor, DBoW2::FORB> voc_txt;
//   voc_txt.loadFromTextFile("../../support_files/vocabulary/ORBvoc.txt");
//
//   testDatabaseVoc(voc_bin, features);
//   LOG(INFO) << "=======";
//   testDatabaseVoc(voc_txt, features);
//   LOG(INFO) << "+++++++";
//   testDatabase(features);
// }

// TEST(DBoW2Test, LoadTxtTest) {
//   VocabularyBinary<DBoW2::FORB::TDescriptor, DBoW2::FORB> voc;
//   voc.loadFromTextFile("../../support_files/vocabulary/ORBvoc.txt");
//   voc.saveToBinaryFile("../../support_files/vocabulary/ORBvoc.bin");
// }
//
// TEST(DBoW2Test, LoadBinTest) {
//   VocabularyBinary<DBoW2::FORB::TDescriptor, DBoW2::FORB> voc;
//   voc.loadFromBinaryFile("../../support_files/vocabulary/ORBvoc.bin");
//   LOG(INFO) << "Binary loaded!";
// }

// TEST(DBoW2Test, VocTest) {
//   VocabularyBinary<DBoW2::FORB::TDescriptor, DBoW2::FORB> voc_bin;
//   voc_bin.loadFromBinaryFile("../../support_files/vocabulary/ORBvoc.bin");
//   LOG(INFO) << voc_bin.GetK() << " " << voc_bin.GetL() << " " << voc_bin.GetWeighting() << " " << voc_bin.GetScoring()
//             << " " << voc_bin.GetNodes().size();
//
//   VocabularyBinary<DBoW2::FORB::TDescriptor, DBoW2::FORB> voc_txt;
//   voc_txt.loadFromTextFile("../../support_files/vocabulary/ORBvoc.txt");
//
//   // VocabularyBinaryBoost<DBoW2::FORB::TDescriptor, DBoW2::FORB> voc_txt;
//   // voc_txt.loadFromBinaryFile("../../support_files/vocabulary/ORBvoc.boost.bin");
//
//   EXPECT_EQ(voc_bin.GetK(), voc_txt.GetK());
//   EXPECT_EQ(voc_bin.GetL(), voc_txt.GetL());
//   EXPECT_EQ(voc_bin.GetScoring(), voc_txt.GetScoring());
//   EXPECT_EQ(voc_bin.GetWeighting(), voc_txt.GetWeighting());
//   EXPECT_EQ(voc_bin.GetNodes().size(), voc_txt.GetNodes().size());
//
//   for (int i = 0; i < voc_bin.GetNodes().size(); ++i) {
//     EXPECT_EQ(cv::norm(voc_bin.GetNodes()[i].descriptor - voc_txt.GetNodes()[i].descriptor), 0);
//     EXPECT_EQ(voc_bin.GetNodes()[i].id, voc_txt.GetNodes()[i].id);
//     EXPECT_EQ(voc_bin.GetNodes()[i].word_id, voc_txt.GetNodes()[i].word_id);
//     EXPECT_EQ(voc_bin.GetNodes()[i].parent, voc_txt.GetNodes()[i].parent);
//     EXPECT_NEAR(voc_bin.GetNodes()[i].weight, voc_txt.GetNodes()[i].weight, 1e-6);
//     EXPECT_EQ(voc_bin.GetNodes()[i].isLeaf(), voc_txt.GetNodes()[i].isLeaf());
//     for (int j = 0; j < voc_bin.GetNodes()[i].children.size(); ++j) {
//       EXPECT_EQ(voc_bin.GetNodes()[i].children[j], voc_txt.GetNodes()[i].children[j]);
//     }
//   }
//
// }

// TEST(DBoW2Test, DemoBinTest) {
//   vector<vector<cv::Mat> > features;
//   loadFeatures(features);
//
//   VocabularyBinary<DBoW2::FORB::TDescriptor, DBoW2::FORB> voc_bin;
//   voc_bin.loadFromBinaryFile("../../support_files/vocabulary/ORBvoc.bin");
//   LOG(INFO) << voc_bin.GetK() << " " << voc_bin.GetL() << " " << voc_bin.GetWeighting() << " " << voc_bin.GetScoring()
//             << " " << voc_bin.GetNodes().size();
//
//   VocabularyBinary<DBoW2::FORB::TDescriptor, DBoW2::FORB> voc_txt;
//   voc_txt.loadFromTextFile("../../support_files/vocabulary/ORBvoc.txt");
//
//   testDatabaseVoc(voc_bin, features);
//   LOG(INFO) << "=======";
//   testDatabaseVoc(voc_txt, features);
//   LOG(INFO) << "+++++++";
//   testDatabase(features);
// }

// TEST(DBoW2Test, OrbTest) {
//   VocabularyBinary<DBoW2::FORB::TDescriptor, DBoW2::FORB> voc_bin;
//   voc_bin.loadFromBinaryFile("../../support_files/vocabulary/ORBvoc.bin");
//   LOG(INFO) << voc_bin.GetK() << " " << voc_bin.GetL() << " " << voc_bin.GetWeighting() << " " << voc_bin.GetScoring()
//             << " " << voc_bin.GetNodes().size();
//
//   ORBextractor feature_extractor(500, 1.2, 5, 20, 7);
//   NaiveFeatureExtractor naive_feature_extractor;
//   vector<vector<cv::Mat> > features;
//   loadOrbFeatures(feature_extractor, features);
//   LOG(INFO) << features[0].size();
//
//   testDatabaseVoc(voc_bin, features);
//
//   loadOrbFeatures(naive_feature_extractor, features);
//   LOG(INFO) << features[0].size();
//
//   testDatabaseVoc(voc_bin, features);
// }

// TEST(DBoW2Test, NodeTest) {
//   VocabularyBinary<DBoW2::FORB::TDescriptor, DBoW2::FORB> voc_bin;
//   voc_bin.loadFromBinaryFile("../../support_files/vocabulary/ORBvoc.bin");
//   // voc_bin.loadFromTextFile("../../support_files/vocabulary/ORBvoc.txt");
//   LOG(INFO) << voc_bin.GetK() << " " << voc_bin.GetL() << " " << voc_bin.GetWeighting() << " " << voc_bin.GetScoring()
//             << " " << voc_bin.GetNodes().size() << ", voc size: " << voc_bin.size();
//   auto &&nodes = voc_bin.GetNodes();
//
//   int node_id = 0;
//   int wid = nodes[node_id].word_id;
//   std::vector<NodeId> node_ids = nodes[node_id].children;
//   LOG(INFO) << "node_id: " << node_id << ", wid: " << wid << ", children size: " << node_ids.size()
//             << " weight: " << nodes[node_id].weight << " descriptor: " << nodes[node_id].descriptor;
//   do {
//     node_id = node_ids[0];
//     node_ids = nodes[node_id].children;
//     wid = nodes[node_id].word_id;
//     LOG(INFO) << "node_id: " << node_id << ", wid: " << wid << ", children size: " << node_ids.size()
//               << " weight: " << nodes[node_id].weight << " descriptor: " << nodes[node_id].descriptor;
//   } while (!nodes[node_id].isLeaf());
//
//   node_id = 0;
//   wid = nodes[node_id].word_id;
//   node_ids = nodes[node_id].children;
//   LOG(INFO) << "node_id: " << node_id << ", wid: " << wid << ", children size: " << node_ids.size()
//             << " weight: " << nodes[node_id].weight << " descriptor: " << nodes[node_id].descriptor;
//   do {
//     node_id = node_ids.back();
//     node_ids = nodes[node_id].children;
//     wid = nodes[node_id].word_id;
//     LOG(INFO) << "node_id: " << node_id << ", wid: " << wid << ", children size: " << node_ids.size()
//               << " weight: " << nodes[node_id].weight << " descriptor: " << nodes[node_id].descriptor;
//   } while (!nodes[node_id].isLeaf());
// }

TEST(BoostSerialization, KeyPointTest) {
  std::string filename("/tmp/test.boost.bin");
  // save data to archive
  cv::KeyPoint kpt_out, kpt_in;
  {
    std::ofstream ofs(filename);
    kpt_out.pt.x = 100;
    kpt_out.pt.y = 200;
    kpt_out.size = 0.5;
    kpt_out.octave = 2;
    kpt_out.class_id = 100;
    kpt_out.angle = 0.5;
    kpt_out.response = 1.5;
    boost::archive::binary_oarchive oa(ofs);
    oa << kpt_out;
  }

  {
    std::ifstream ifs(filename);
    boost::archive::binary_iarchive ia(ifs);
    ia >> kpt_in;

    LOG(INFO) << kpt_in.pt << " " << kpt_out.pt;
  }

  {
    EXPECT_EQ(kpt_in.octave, kpt_out.octave);
    EXPECT_NEAR(kpt_in.pt.x, kpt_out.pt.x, 1e-6);
    EXPECT_NEAR(kpt_in.pt.y, kpt_out.pt.y, 1e-6);
    EXPECT_NEAR(kpt_in.size, kpt_out.size, 1e-6);
    EXPECT_EQ(kpt_in.class_id, kpt_out.class_id);
    EXPECT_NEAR(kpt_in.angle, kpt_out.angle, 1e-6);
    EXPECT_NEAR(kpt_in.response, kpt_out.response, 1e-6);
  }
}

TEST(BoostSerialization, DBoW2Test) {
  std::string filename("/tmp/test.boost.bin");
  DBoW2::BowVector bow_vec_out, bow_vec_in;
  DBoW2::FeatureVector feat_vec_out, feat_vec_in;
  {
    std::ofstream ofs(filename);
    boost::archive::binary_oarchive oa(ofs);

    bow_vec_out.insert(std::pair<unsigned int, double>(1, 1.5));
    bow_vec_out.insert(std::pair<unsigned int, double>(2, 2.5));
    feat_vec_out.insert(std::pair<unsigned int, std::vector<unsigned int>>(3, std::vector<unsigned int>{3, 4}));
    feat_vec_out.insert(std::pair<unsigned int, std::vector<unsigned int>>(4, std::vector<unsigned int>{1, 2}));

    oa << bow_vec_out;
    oa << feat_vec_out;
  }

  {
    std::ifstream ifs(filename);
    boost::archive::binary_iarchive ia(ifs);
    ia >> bow_vec_in;
    ia >> feat_vec_in;
  }

  {

    for (auto&& pair : bow_vec_out) {
      EXPECT_NEAR(bow_vec_in[pair.first], bow_vec_out[pair.first], 1e-6);
      LOG(INFO) << bow_vec_in[pair.first] << " " << bow_vec_out[pair.first];
    }


    for (auto&& pair : feat_vec_out) {
      for (int i = 0; i < feat_vec_out[pair.first].size(); ++i) {
        LOG(INFO) << feat_vec_in[pair.first][i] << " " << feat_vec_out[pair.first][i];
        EXPECT_EQ(feat_vec_in[pair.first][i], feat_vec_out[pair.first][i]);
      }
    }
  }
}

void loadOrbFeatures(FeatureExtractor &extractor, vector<vector<cv::Mat> > &features) {
  features.clear();
  features.reserve(NIMAGES);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << "Extracting ORB features..." << endl;
  for (int i = 0; i < NIMAGES; ++i) {
    stringstream ss;
    ss << "../../3rdparty/DBoW2/demo/images/image" << i << ".png";

    cv::Mat image = cv::imread(ss.str(), 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    extractor(image, mask, keypoints, descriptors);

    features.push_back(vector<cv::Mat>());
    changeStructure(descriptors, features.back());
  }
}

void loadFeatures(vector<vector<cv::Mat> > &features) {
  features.clear();
  features.reserve(NIMAGES);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << "Extracting ORB features..." << endl;
  for (int i = 0; i < NIMAGES; ++i) {
    stringstream ss;
    ss << "../../3rdparty/DBoW2/demo/images/image" << i << ".png";

    cv::Mat image = cv::imread(ss.str(), 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    features.push_back(vector<cv::Mat>());
    changeStructure(descriptors, features.back());
  }
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out) {
  out.resize(plain.rows);

  for (int i = 0; i < plain.rows; ++i) {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<cv::Mat> > &features) {
  // branching factor and depth levels
  const int k = 9;
  const int L = 3;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L1_NORM;

  OrbVocabulary voc(k, L, weight, scoring);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
       << voc << endl << endl;

  // lets do something with this vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  for (int i = 0; i < NIMAGES; i++) {
    voc.transform(features[i], v1);
    for (int j = 0; j < NIMAGES; j++) {
      voc.transform(features[j], v2);

      double score = voc.score(v1, v2);
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("small_voc.yml.gz");
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testDatabase(const vector<vector<cv::Mat> > &features) {
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  OrbVocabulary voc("small_voc.yml.gz");

  testDatabaseVoc(voc, features);
}

template<class TDescriptor, class F>
void testDatabaseVoc(const TemplatedVocabulary<TDescriptor, F> &voc, const vector<vector<cv::Mat> > &features) {
  TemplatedDatabase<TDescriptor, F> db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for (int i = 0; i < NIMAGES; i++) {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for (int i = 0; i < NIMAGES; i++) {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save("small_db.yml.gz");
  cout << "... done!" << endl;

  // once saved, we can load it again
  cout << "Retrieving database once again..." << endl;
  OrbDatabase db2("small_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;

  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}