#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void GroupingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 2);
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
}

template <typename Dtype>
void GroupingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  group_channels_ = bottom[1]->channels();
  top[0]->Reshape(num_, channels_, height_, width_);
  const Dtype* group_data = bottom[1]->cpu_data();

  // map group_data to [0, num_group - 1] and store it at group_blob_
  group_blob_.ReshapeLike(*bottom[1]);
  Dtype* group_blob_data = group_blob_.mutable_cpu_data();
  for (int n = 0; n < num_; ++n) {
    for (int c = 0; c < group_channels_; ++c) {
      map<int, int> group_id_map;
      group_id_map.clear();
      int count = -1;
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          int index = ((n * group_channels_ + c) * height_ + h) * width_ + w;
          int group_id = group_data[index];
          if (group_id_map.find(group_id) == group_id_map.end()) {
            ++count;
            group_id_map[group_id] = count;
            group_blob_data[index] = count;
          } else {
            group_blob_data[index] = group_id_map[group_id];
          }
        }
      }
    }
  }
  // get the map from pixel index to group index
  group_maps_vec_.clear();
  for (int n = 0; n < num_; ++n) {
    vector<map<int, vector<int> > > group_maps;
    group_maps.clear();
    for (int gc = 0; gc < group_channels_; ++gc) {
      map<int, vector<int> > group_map;
      group_map.clear();
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          int index = h * width_ + w;
          int group_index = (int)group_blob_data[index];
          group_map[group_index].push_back(index);
        }
      }
      group_blob_data += group_blob_.offset(0, 1);
      group_maps.push_back(group_map);
    }
    group_maps_vec_.push_back(group_maps);
  }
}

template <typename Dtype>
void GroupingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  caffe_set(top_count, Dtype(0), top_data);
  // the main loop
  for (int n = 0; n < num_; ++n) {
    vector<map<int, vector<int> > >& group_maps = group_maps_vec_[n];
    for (int gc = 0; gc < group_channels_; ++gc) {
      map<int, vector<int> >& group_map = group_maps[gc];
      group_mean_.Reshape(1, 1, 1, group_map.size());
      Dtype* group_mean_data = group_mean_.mutable_cpu_data();
      // reset bottom_data and top_data
      bottom_data = bottom[0]->cpu_data() + bottom[0]->offset(n);
      top_data = top[0]->mutable_cpu_data() + top[0]->offset(n);
      for (int c = 0; c < channels_; ++c) {
        caffe_set(group_mean_.count(), Dtype(0), group_mean_data);
        // compute the mean for each group
        int group_id = 0;
        for (map<int, vector<int> >::iterator it = group_map.begin();
             it != group_map.end(); ++it) {
          for (int s = 0; s < it->second.size(); ++s) {
            group_mean_data[group_id] += bottom_data[it->second[s]];
          }
          group_mean_data[group_id] /= it->second.size() * bottom[1]->channels();
          ++group_id;
        }
        // assign the mean to top_data
        group_id = 0;
        for (map<int, vector<int> >::iterator it = group_map.begin();
             it != group_map.end(); ++it) {
          for (int s = 0; s < it->second.size(); ++s) {
            top_data[it->second[s]] += group_mean_data[group_id];
          }
          ++group_id;
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
  }
}

template <typename Dtype>
void GroupingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // the main loop
  for (int n = 0; n < num_; ++n) {
    vector<map<int, vector<int> > >& group_maps = group_maps_vec_[n];
    for (int c = 0; c < channels_; ++c) {
      for (int gc = 0; gc < group_channels_; ++gc) {
        map<int, vector<int> >& group_map = group_maps[gc];
        for (map<int, vector<int> >::iterator it = group_map.begin();
             it != group_map.end(); ++it) {
          for (int s1 = 0; s1 < it->second.size(); ++s1) {
            int index1 = it->second[s1];
            for (int s2 = 0; s2 < it->second.size(); ++s2) {
              int index2 = it->second[s2];
              bottom_diff[index1] +=
                  top_diff[index2] / it->second.size() / group_channels_;
            }
          }
        }
      }
      // compute offset
      bottom_diff += bottom[0]->offset(0, 1);
      top_diff += top[0]->offset(0, 1);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(GroupingLayer);
#endif

INSTANTIATE_CLASS(GroupingLayer);
REGISTER_LAYER_CLASS(Grouping);

}  // namespace caffe
