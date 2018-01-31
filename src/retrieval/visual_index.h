// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_RETRIEVAL_VISUAL_INDEX_H_
#define COLMAP_SRC_RETRIEVAL_VISUAL_INDEX_H_

#include <Eigen/Core>

#include "ext/FLANN/flann.hpp"
#include "feature/types.h"
#include "retrieval/inverted_file.h"
#include "retrieval/inverted_index.h"
#include "retrieval/vote_and_verify.h"
#include "util/alignment.h"
#include "util/endian.h"
#include "util/logging.h"
#include "util/math.h"

namespace colmap {
namespace retrieval {

// Visual index for image retrieval using a vocabulary tree with Hamming
// embedding, based on the papers:
//
//    Schönberger, Price, Sattler, Pollefeys, Frahm. "A Vote-and-Verify Strategy
//    for Fast Spatial Verification in Image Retrieval". ACCV 2016.
//
//    Arandjelovic, Zisserman: Scalable descriptor
//    distinctiveness for location recognition. ACCV 2014.
template <typename kDescType = uint8_t, int kDescDim = 128,
          int kEmbeddingDim = 64>
class VisualIndex {
 public:
  static const int kMaxNumThreads = -1;
  typedef InvertedIndex<kDescType, kDescDim, kEmbeddingDim> InvertedIndexType;
  typedef FeatureKeypoints GeomType;
  typedef typename InvertedIndexType::DescType DescType;
  typedef Eigen::VectorXf ProjDescType;

  struct IndexOptions {
    // The number of nearest neighbor visual words that each feature descriptor
    // is assigned to.
    int num_neighbors = 1;

    // The number of checks in the nearest neighbor search.
    int num_checks = 256;

    // The number of threads used in the index.
    int num_threads = kMaxNumThreads;
  };

  struct QueryOptions {
    // The maximum number of most similar images to retrieve.
    int max_num_images = -1;

    // The number of images to be spatially verified and reranked.
    int max_num_verifications = -1;

    // The number of nearest neighbor visual words that each feature descriptor
    // is assigned to.
    int num_neighbors = 5;

    // The number of checks in the nearest neighbor search.
    int num_checks = 256;

    // The number of threads used in the index.
    int num_threads = kMaxNumThreads;
  };

  struct BuildOptions {
    // The desired number of visual words, i.e. the number of leaf node
    // clusters. Note that the actual number of visual words might be less.
    int num_visual_words = 256 * 256;

    // The branching factor of the hierarchical k-means tree.
    int branching = 256;

    // The number of iterations for the clustering.
    int num_iterations = 11;

    // The target precision of the visual word search index.
    double target_precision = 0.9;

    // The number of checks in the nearest neighbor search.
    int num_checks = 256;

    // The number of threads used in the index.
    int num_threads = kMaxNumThreads;
  };

  VisualIndex();
  ~VisualIndex();

  size_t NumVisualWords() const;

  // Add image to the visual index.
  void Add(const IndexOptions& options, const int image_id,
           const GeomType& geometries, const DescType& descriptors);

  // Query for most similar images in the visual index.
  void Query(const QueryOptions& options, const DescType& descriptors,
             std::vector<ImageScore>* image_scores) const;

  // Query for most similar images in the visual index.
  void QueryWithVerification(const QueryOptions& options,
                             const GeomType& geometries,
                             const DescType& descriptors,
                             std::vector<ImageScore>* image_scores) const;

  // Given a set of matches for a database images, where each match is given as
  // indices, returns a set of 1-to-1 correspondences suitable for spatial
  // verification. To select 1-to-1 matches, the approach from Li et al.,
  // Pairwise Geometric Matching for Large-scale Object Retrieval. CVPR 2015
  // is used (cf. Sec. 4.1 in the paper). Returns the number of matches.
  int Get1To1MatchesForSpatialVerification(
      const GeomType& geometries_query,
      const std::vector<FeatureGeometryMatch>& initial_matches,
      std::vector<FeatureGeometryMatch> *matches_1to1) const;

  // Prepare the index after adding images and before querying.
  void Prepare();

  // Build a visual index from a set of training descriptors by quantizing the
  // descriptor space into visual words and compute their Hamming embedding.
  void Build(const BuildOptions& options, const DescType& descriptors);

  // Read and write the visual index. This can be done for an index with and
  // without indexed images.
  void Read(const std::string& path);
  void Write(const std::string& path);

 private:
  // Quantize the descriptor space into visual words.
  void Quantize(const BuildOptions& options, const DescType& descriptors);

  // Query for nearest neighbor images and return nearest neighbor visual word
  // identifiers for each descriptor.
  void QueryAndFindWordIds(const QueryOptions& options,
                           const DescType& descriptors,
                           std::vector<ImageScore>* image_scores,
                           Eigen::MatrixXi* word_ids) const;

  // Find the nearest neighbor visual words for the given descriptors.
  Eigen::MatrixXi FindWordIds(const DescType& descriptors,
                              const int num_neighbors, const int num_checks,
                              const int num_threads) const;

  // Given an unordered map of (id, value) pairs, returns the id and value of
  // the pair with lowest value. id is -1 if no such element can be found.
  void GetIdAndValueForLowestValue(const std::unordered_map<int, int>& map,
                                   int* id, int *value) const;


  // Maps a word - feature index pair to a string that can be used for indexing.
  void WordFeatureIndexPairToString(int word_id, int feature_index,
                                    std::string* s) const;


  // The search structure on the quantized descriptor space.
  flann::AutotunedIndex<flann::L2<kDescType>> visual_word_index_;

  // The centroids of the visual words.
  flann::Matrix<kDescType> visual_words_;

  // The inverted index of the database.
  InvertedIndexType inverted_index_;

  // Identifiers of all indexed images.
  std::unordered_set<int> image_ids_;

  // Whether the index is prepared.
  bool prepared_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

  template <typename kDescType, int kDescDim, int kEmbeddingDim>
  void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::GetIdAndValueForLowestValue(const std::unordered_map<int, int>& map,
                                   int* id, int *value) const {
    *id = -1;
    int val = std::numeric_limits<int>::max();
    for (const std::pair<int, int>& it : map) {
      if (it.second < val) {
        val = it.second;
        *id = it.first;
      }
    }

    *value = val;
  }

  template <typename kDescType, int kDescDim, int kEmbeddingDim>
  void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::WordFeatureIndexPairToString(int word_id, int feature_index,
                                    std::string* s) const {
    std::stringstream s_stream;
    s_stream << word_id << "_" << feature_index;
    *s = s_stream.str();
  }

  template <typename kDescType, int kDescDim, int kEmbeddingDim>
  int VisualIndex<kDescType, kDescDim, kEmbeddingDim>::Get1To1MatchesForSpatialVerification(
      const GeomType& geometries_query,
      const std::vector<FeatureGeometryMatch>& initial_matches,
      std::vector<FeatureGeometryMatch> *matches_1to1) const {
    matches_1to1->clear();

    if (initial_matches.size() < 4u) return 0;

    typedef std::pair<int, float> WeightedMatch;
    typedef std::vector<WeightedMatch> WeightedMatchVec;

    // Stores for each query feature a vector of relevant database features.
    // Stores for each database feature a vector  of relevant query features and
    // assigns it a unique id. Also, stores for each database feature its
    // affine parameters.
    std::unordered_map<int, WeightedMatchVec> matches_per_query;
    std::unordered_map<std::string, int> database_ids;
    std::unordered_map<int, WeightedMatchVec> matches_per_db;
    std::unordered_map<int, FeatureGeometry> db_feat_geometries;

    int current_db_feature_index = 0;
    for (const FeatureGeometryMatch& init_match : initial_matches) {
      // Get the id of the database feature.
      std::string word_id_string;
      for (int i = 0; i < init_match.geometries2.size(); i++){
        int db_feature_word = init_match.word_id;
        int db_feature_index = i;
        WordFeatureIndexPairToString(db_feature_word,
                                   db_feature_index,
                                   &word_id_string);
        int db_feat_id = database_ids.emplace(word_id_string, current_db_feature_index).first->second;

        db_feat_geometries[db_feat_id] = init_match.geometries2[i];

        // Updates the matches found for the query and database images.
        matches_per_query[init_match.feature_id].push_back(
            std::make_pair(db_feat_id, init_match.match_weights[i]));

        matches_per_db[db_feat_id].push_back(
            std::make_pair(init_match.feature_id, init_match.match_weights[i]));

        ++current_db_feature_index;
      }
    }

    // Stores for each feature in each image the number of possible matches from
    // which we can choose. At the same time, populates two vectors containing
    // a pair (feature_id, num_matches) for each feature.
    std::unordered_map<int, int> num_matches_per_query;
    std::unordered_map<int, bool> query_feature_selected;

    std::unordered_map<int, int> num_matches_per_db;
    std::unordered_map<int, bool> db_feature_selected;

    for (const std::pair<int, WeightedMatchVec>& m : matches_per_query) {
      int num_potential_matches = static_cast<int>(m.second.size());
      num_matches_per_query[m.first] = num_potential_matches;
      query_feature_selected[m.first] = false;
    }
    for (const std::pair<int, WeightedMatchVec>& m : matches_per_db) {
      int num_potential_matches = static_cast<int>(m.second.size());
      num_matches_per_db[m.first] = num_potential_matches;
      db_feature_selected[m.first] = false;
    }

    // Actually selects the matches.
    // TODO(sattler): Think about a faster implementation!
    int num_matches = 0;
    int best_query_id = -1;
    int best_query_value = -1;
    int best_db_id = -1;
    int best_db_value = -1;
    while (true) {
      // Obtains the two features (one in the query image, the other in the
      // database image) for which the smallest number of potential matches
      // exists and selects the one contained in the lowest number of matches.
      GetIdAndValueForLowestValue(num_matches_per_query, &best_query_id, &best_query_value);
      GetIdAndValueForLowestValue(num_matches_per_db, &best_db_id, &best_db_value);

      if (best_query_id == -1 && best_db_id == -1) {
        // No more matches can be found.
        break;
      }

      bool use_query = best_query_value < best_db_value;
      if (best_query_value == -1) use_query = false;

      // Finds the best match with the highest similarity score.
      int best_matching_feature = -1;
      float best_matching_score = -1.0f;
      const WeightedMatchVec& vec =
          use_query ?
              matches_per_query[best_query_id] : matches_per_db[best_db_id];
      const std::unordered_map<int, bool>& feat_selected =
          use_query ? db_feature_selected : query_feature_selected;
      for (const WeightedMatch& m : vec) {
        if ((m.second > best_matching_score)
            && (feat_selected.at(m.first) == false)) {
          best_matching_score = m.second;
          best_matching_feature = m.first;
        }
      }

      // Adds the match.
      ++num_matches;
      FeatureGeometryMatch new_match;
      const int f1_id = use_query? best_query_id : best_matching_feature;
      new_match.feature_id = f1_id;
      new_match.geometry1.x = geometries_query[f1_id].x;
      new_match.geometry1.y = geometries_query[f1_id].y;
      new_match.geometry1.scale = geometries_query[f1_id].ComputeScale();
      new_match.geometry1.orientation =
          geometries_query[f1_id].ComputeOrientation();

      const int f2_id = use_query? best_matching_feature : best_db_id;
      new_match.geometries2.push_back(db_feat_geometries[f2_id]);
      matches_1to1->push_back(new_match);

//      std::cout << " selected: " << use_query << " " << f1_id << " " << f2_id << std::endl;
//      std::cout << db_feat_geometries[f2_id].x_.transpose() << " " << db_feat_geometries[f2_id].a_ << " " << db_feat_geometries[f2_id].b_ << " " << db_feat_geometries[f2_id].c_ << std::endl;

      // Updates the list of potential matches.
      query_feature_selected[f1_id] = true;
      db_feature_selected[f2_id] = true;
      num_matches_per_query.erase(f1_id);
      num_matches_per_db.erase(f2_id);
      for (const WeightedMatch& m : matches_per_query[f1_id]) {
        std::unordered_map<int, int>::iterator it = num_matches_per_db.find(
            m.first);
        if (it != num_matches_per_db.end()) {
          it->second -= 1;
          if (it->second <= 0) num_matches_per_db.erase(it->first);
        }
      }
      for (const WeightedMatch& m : matches_per_db[f2_id]) {
        std::unordered_map<int, int>::iterator it = num_matches_per_query.find(
            m.first);
        if (it != num_matches_per_query.end()) {
          it->second -= 1;
          if (it->second <= 0) num_matches_per_query.erase(it->first);
        }
      }
    }

    return num_matches;
  }

template <typename kDescType, int kDescDim, int kEmbeddingDim>
VisualIndex<kDescType, kDescDim, kEmbeddingDim>::VisualIndex()
    : prepared_(false) {}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
VisualIndex<kDescType, kDescDim, kEmbeddingDim>::~VisualIndex() {
  if (visual_words_.ptr() != nullptr) {
    delete[] visual_words_.ptr();
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
size_t VisualIndex<kDescType, kDescDim, kEmbeddingDim>::NumVisualWords() const {
  return visual_words_.rows;
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::Add(
    const IndexOptions& options, const int image_id, const GeomType& geometries,
    const DescType& descriptors) {
  CHECK(image_ids_.count(image_id) == 0);
  image_ids_.insert(image_id);

  prepared_ = false;

  if (descriptors.rows() == 0) {
    return;
  }

  const Eigen::MatrixXi word_ids =
      FindWordIds(descriptors, options.num_neighbors, options.num_checks,
                  options.num_threads);

  for (typename DescType::Index i = 0; i < descriptors.rows(); ++i) {
    const auto& descriptor = descriptors.row(i);

    typename InvertedIndexType::GeomType geometry;
    geometry.x = geometries[i].x;
    geometry.y = geometries[i].y;
    geometry.scale = geometries[i].ComputeScale();
    geometry.orientation = geometries[i].ComputeOrientation();

    for (int n = 0; n < options.num_neighbors; ++n) {
      const int word_id = word_ids(i, n);
      if (word_id != InvertedIndexType::kInvalidWordId) {
        inverted_index_.AddEntry(image_id, word_id, descriptor, geometry);
      }
    }
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::Query(
    const QueryOptions& options, const DescType& descriptors,
    std::vector<ImageScore>* image_scores) const {
  Eigen::MatrixXi word_ids;
  QueryAndFindWordIds(options, descriptors, image_scores, &word_ids);
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::QueryWithVerification(
    const QueryOptions& options, const GeomType& geometries_query,
    const DescType& descriptors, std::vector<ImageScore>* image_scores) const {
  // geometries_query contain keypoint information.
  CHECK_EQ(descriptors.rows(), geometries_query.size());

  size_t num_verifications = image_ids_.size();
  if (options.max_num_verifications >= 0) {
    num_verifications =
        std::min<size_t>(image_ids_.size(), options.max_num_verifications);
  }

  if (num_verifications == 0) {
    Query(options, descriptors, image_scores);
    return;
  }


  auto verification_options = options;
  verification_options.max_num_images = options.max_num_verifications;

  Eigen::MatrixXi word_ids;
  QueryAndFindWordIds(verification_options, descriptors, image_scores,
                      &word_ids);

  // Extract top-ranked images to verify.
  std::unordered_set<int> image_ids; // keep only images for spatial verification.
  for (size_t i = 0; i < num_verifications; ++i) {
    image_ids.insert((*image_scores)[i].image_id);
  }

  // Find matches for top-ranked images, only use single nearest neighbor word.
  std::vector<std::tuple<int, typename InvertedIndexType::GeomType, float>>
      word_matches; // It conatins tuple of (image_id, keypoint, correspondence_weight)
  std::unordered_map<int, std::unordered_map<int, std::tuple<std::vector<FeatureGeometry>, std::vector<float>, int>>>
      image_matches;
  for (typename DescType::Index i = 0; i < descriptors.rows(); ++i) {
    const int word_id = word_ids(i, 0); // assigned word_id for i-th descriptor
    if (word_id != InvertedIndexType::kInvalidWordId) {
      // Inverted index is a dictionary with key: word_id, value: list of pair:(image_id, keypoint)
      // FindMatches find every keypoint match on every entry.
      // passed image_ids work as whiltelist filter.
      const ProjDescType proj_descriptor =
        inverted_index_.GetProjectionMatrix() * descriptors.row(i).transpose().template cast<float>();
      // inverted_index_.FindMatches(word_id, image_ids, &word_matches); // Find all matches
      inverted_index_.FindMatchesWithWeight(proj_descriptor, word_id, image_ids, &word_matches); // Find all matches
      for (const auto& match : word_matches) {
        int image_id_idx = std::get<0>(match); // index of matched image_id
        auto keypoint = std::get<1>(match);
        float weight = std::get<2>(match);

        // image_matches[image_id that contains matched keypoint][i-th descriptor] save keypoint for the matched image.

        std::get<0>(image_matches[image_id_idx][i]).emplace_back(keypoint);
        std::get<1>(image_matches[image_id_idx][i]).emplace_back(weight);
        std::get<2>(image_matches[image_id_idx][i]) = word_id;
      }
    }
  }

  // Verify top-ranked images using the found matches.
  for (size_t i = 0; i < num_verifications; ++i) {
    auto& image_score = (*image_scores)[i];
    const std::unordered_map<int, std::tuple<std::vector<FeatureGeometry>, std::vector<float>, int>>& geometry_matches = image_matches[image_score.image_id]; // all matches with i-th image

    // No matches found,
    if (geometry_matches.empty()) {
      continue;
    }

    // Collect matches for all features of current image.
    std::vector<FeatureGeometryMatch> matches;
    matches.reserve(geometry_matches.size());
    for (const auto& geometries2 : geometry_matches) { // iterate over unordered_map
      // geometries2.first: descriptor index.
      auto& query_descriptor_index = geometries2.first;
      // geometries2.second: tuple of <list of matched keypoint of image, list of match weight>
      auto& tup = geometries2.second;
      auto& matched_keypoints = std::get<0>(tup);
      auto& match_weights = std::get<1>(tup);
      auto& word_id = std::get<2>(tup);
      FeatureGeometryMatch match;
      match.feature_id = query_descriptor_index;
      match.word_id = word_id;
      match.geometry1.x = geometries_query[query_descriptor_index].x;
      match.geometry1.y = geometries_query[query_descriptor_index].y;
      match.geometry1.scale = geometries_query[query_descriptor_index].ComputeScale();
      match.geometry1.orientation =
          geometries_query[query_descriptor_index].ComputeOrientation();
      match.geometries2 = matched_keypoints;
      match.match_weights = match_weights;
      matches.push_back(match);
    }

    std::vector<FeatureGeometryMatch> matches_1to1;
    Get1To1MatchesForSpatialVerification(geometries_query, matches, &matches_1to1);

    VoteAndVerifyOptions vote_and_verify_options;
    auto geom_verif_score = VoteAndVerify(vote_and_verify_options, matches_1to1);
    image_score.score += geom_verif_score;
  }

  // Re-rank the images using the spatial verification scores.

  size_t num_images = image_scores->size();
  if (options.max_num_images >= 0) {
    num_images = std::min<size_t>(image_scores->size(), options.max_num_images);
  }

  auto SortFunc = [](const ImageScore& score1, const ImageScore& score2) {
    return score1.score > score2.score;
  };

  if (num_images == image_scores->size()) {
    std::sort(image_scores->begin(), image_scores->end(), SortFunc);
  } else {
    // when num_images < image_scores->size()
    std::partial_sort(image_scores->begin(), image_scores->begin() + num_images,
                      image_scores->end(), SortFunc);
    image_scores->resize(num_images);
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::Prepare() {
  inverted_index_.Finalize();
  prepared_ = true;
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::Build(
    const BuildOptions& options, const DescType& descriptors) {
  // Quantize the descriptor space into visual words.
  Quantize(options, descriptors);

  // Build the search index on the visual words.
  flann::AutotunedIndexParams index_params;
  index_params["target_precision"] =
      static_cast<float>(options.target_precision);
  visual_word_index_ =
      flann::AutotunedIndex<flann::L2<kDescType>>(index_params);
  visual_word_index_.buildIndex(visual_words_);

  // Initialize a new inverted index.
  inverted_index_ = InvertedIndexType();
  inverted_index_.Initialize(NumVisualWords());

  // Generate descriptor projection matrix.
  inverted_index_.GenerateHammingEmbeddingProjection();

  // Learn the Hamming embedding.
  const int kNumNeighbors = 1;
  const Eigen::MatrixXi word_ids = FindWordIds(
      descriptors, kNumNeighbors, options.num_checks, options.num_threads);
  inverted_index_.ComputeHammingEmbedding(descriptors, word_ids);
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::Read(
    const std::string& path) {
  long int file_offset = 0;

  // Read the visual words.

  {
    if (visual_words_.ptr() != nullptr) {
      delete[] visual_words_.ptr();
    }

    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;
    const uint64_t rows = ReadBinaryLittleEndian<uint64_t>(&file);
    const uint64_t cols = ReadBinaryLittleEndian<uint64_t>(&file);
    kDescType* visual_words_data = new kDescType[rows * cols];
    for (size_t i = 0; i < rows * cols; ++i) {
      visual_words_data[i] = ReadBinaryLittleEndian<kDescType>(&file);
    }
    visual_words_ = flann::Matrix<kDescType>(visual_words_data, rows, cols);
    file_offset = file.tellg();
  }

  // Read the visual words search index.

  visual_word_index_ =
      flann::AutotunedIndex<flann::L2<kDescType>>(visual_words_);

  {
    FILE* fin = fopen(path.c_str(), "rb");
    CHECK_NOTNULL(fin);
    fseek(fin, file_offset, SEEK_SET);
    visual_word_index_.loadIndex(fin);
    file_offset = ftell(fin);
    fclose(fin);
  }

  // Read the inverted index.

  {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;
    file.seekg(file_offset, std::ios::beg);
    inverted_index_.Read(&file);
    file_offset = file.tellg();
  }

  // Read the identifiers of all indexed images.

  {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;
    file.seekg(file_offset, std::ios::beg);
    const size_t size = ReadBinaryLittleEndian<size_t>(&file);
    for (size_t i = 0; i < size; ++i) {
      auto id = ReadBinaryLittleEndian<int>(&file);
      image_ids_.insert(id);
    }
  }

}


template <typename kDescType, int kDescDim, int kEmbeddingDim>
void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::Write(
    const std::string& path) {
  // Write the visual words.

  {
    CHECK_NOTNULL(visual_words_.ptr());
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;
    WriteBinaryLittleEndian<uint64_t>(&file, visual_words_.rows);
    WriteBinaryLittleEndian<uint64_t>(&file, visual_words_.cols);
    for (size_t i = 0; i < visual_words_.rows * visual_words_.cols; ++i) {
      WriteBinaryLittleEndian<kDescType>(&file, visual_words_.ptr()[i]);
    }
  }

  // Write the visual words search index.

  {
    FILE* fout = fopen(path.c_str(), "ab");
    CHECK_NOTNULL(fout);
    visual_word_index_.saveIndex(fout);
    fclose(fout);
  }

  // Write the inverted index.

  {
    std::ofstream file(path, std::ios::binary | std::ios::app);
    CHECK(file.is_open()) << path;
    inverted_index_.Write(&file);
  }

  // Write the identifiers of all indexed images.

  {
    std::ofstream file(path, std::ios::binary | std::ios::app);
    CHECK(file.is_open()) << path;
    WriteBinaryLittleEndian<size_t>(&file, image_ids_.size());
    for (const auto& id: image_ids_) {
        WriteBinaryLittleEndian<int>(&file, id);
    }
  }

}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::Quantize(
    const BuildOptions& options, const DescType& descriptors) {
  static_assert(DescType::IsRowMajor, "Descriptors must be row-major.");

  CHECK_GE(options.num_visual_words, options.branching);
  CHECK_GE(descriptors.rows(), options.num_visual_words);

  const flann::Matrix<kDescType> descriptor_matrix(
      const_cast<kDescType*>(descriptors.data()), descriptors.rows(),
      descriptors.cols());

  std::vector<typename flann::L2<kDescType>::ResultType> centers_data(
      options.num_visual_words * descriptors.cols());
  flann::Matrix<typename flann::L2<kDescType>::ResultType> centers(
      centers_data.data(), options.num_visual_words, descriptors.cols());

  flann::KMeansIndexParams index_params;
  index_params["branching"] = options.branching;
  index_params["iterations"] = options.num_iterations;
  index_params["centers_init"] = flann::FLANN_CENTERS_KMEANSPP;
  const int num_centers = flann::hierarchicalClustering<flann::L2<kDescType>>(
      descriptor_matrix, centers, index_params);

  CHECK_LE(num_centers, options.num_visual_words);

  const size_t visual_word_data_size = num_centers * descriptors.cols();
  kDescType* visual_words_data = new kDescType[visual_word_data_size];
  for (size_t i = 0; i < visual_word_data_size; ++i) {
    if (std::is_integral<kDescType>::value) {
      visual_words_data[i] = std::round(centers_data[i]);
    } else {
      visual_words_data[i] = centers_data[i];
    }
  }

  if (visual_words_.ptr() != nullptr) {
    delete[] visual_words_.ptr();
  }

  visual_words_ = flann::Matrix<kDescType>(visual_words_data, num_centers,
                                           descriptors.cols());
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::QueryAndFindWordIds(
    const QueryOptions& options, const DescType& descriptors,
    std::vector<ImageScore>* image_scores, Eigen::MatrixXi* word_ids) const {
  CHECK(prepared_);

  if (descriptors.rows() == 0) {
    image_scores->clear();
    return;
  }

  *word_ids = FindWordIds(descriptors, options.num_neighbors,
                          options.num_checks, options.num_threads);
  inverted_index_.Query(descriptors, *word_ids, image_scores);

  auto SortFunc = [](const ImageScore& score1, const ImageScore& score2) {
    return score1.score > score2.score;
  };

  size_t num_images = image_scores->size();
  if (options.max_num_images >= 0) {
    num_images = std::min<size_t>(image_scores->size(), options.max_num_images);
  }

  if (num_images == image_scores->size()) {
    std::sort(image_scores->begin(), image_scores->end(), SortFunc);
  } else {
    std::partial_sort(image_scores->begin(), image_scores->begin() + num_images,
                      image_scores->end(), SortFunc);
    image_scores->resize(num_images);
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
Eigen::MatrixXi VisualIndex<kDescType, kDescDim, kEmbeddingDim>::FindWordIds(
    const DescType& descriptors, const int num_neighbors, const int num_checks,
    const int num_threads) const {
  static_assert(DescType::IsRowMajor, "Descriptors must be row-major");

  CHECK_GT(descriptors.rows(), 0);
  CHECK_GT(num_neighbors, 0);

  Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      word_ids(descriptors.rows(), num_neighbors);
  word_ids.setConstant(InvertedIndexType::kInvalidWordId);
  flann::Matrix<size_t> indices(word_ids.data(), descriptors.rows(),
                                num_neighbors);

  Eigen::Matrix<typename flann::L2<kDescType>::ResultType, Eigen::Dynamic,
                Eigen::Dynamic, Eigen::RowMajor>
      distance_matrix(descriptors.rows(), num_neighbors);
  flann::Matrix<typename flann::L2<kDescType>::ResultType> distances(
      distance_matrix.data(), descriptors.rows(), num_neighbors);

  const flann::Matrix<kDescType> query(
      const_cast<kDescType*>(descriptors.data()), descriptors.rows(),
      descriptors.cols());

  flann::SearchParams search_params(num_checks);
  if (num_threads < 0) {
    search_params.cores = std::thread::hardware_concurrency();
  } else {
    search_params.cores = num_threads;
  }
  if (search_params.cores <= 0) {
    search_params.cores = 1;
  }

  visual_word_index_.knnSearch(query, indices, distances, num_neighbors,
                               search_params);

  return word_ids.cast<int>();
}

}  // namespace retrieval
}  // namespace colmap

#endif  // COLMAP_SRC_RETRIEVAL_VISUAL_INDEX_H_
