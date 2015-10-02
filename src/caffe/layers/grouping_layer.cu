#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// convert map to Blob because CUDA code cannot access map STL
void GetMapData(const map<int, vector<int> >& group_map,
    Blob<int>* group_map_range, Blob<int>* group_map_index = NULL) {
  // get the start and end index of all groups
  group_map_range->Reshape(1, 1, 1, group_map.size()+1);
  int* group_map_range_data = group_map_range->mutable_cpu_data();
  int total_count = 0;
  int count = 0;
  for (map<int, vector<int> >::const_iterator it = group_map.begin();
      it != group_map.end(); ++it) {
    group_map_range_data[count] = total_count;
    total_count += it->second.size();
    ++count;
  }
  group_map_range_data[count] = total_count;
  // get the group_map_index if necessary
  if (group_map_index != NULL) {
    group_map_index->Reshape(1, 1, 1, total_count);
    int* group_map_index_data = group_map_index->mutable_cpu_data();
    count = 0;
    for (map<int, vector<int> >::const_iterator it = group_map.begin();
        it != group_map.end(); ++it) {
      for (int s = 0; s < it->second.size(); ++s) {
        group_map_index_data[count] = it->second[s];
        ++count;
      }
    }
  }
}

template <typename Dtype>
__global__ void ComputeGroupMean(const int nthreads, const Dtype* data,
    const int channels, const int height, const int width, const int num_groups,
    const int* group_map_range, const int* group_map_index,
    const int group_channels, Dtype* group_mean_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int group_id = index % num_groups;
    int c = (index / num_groups) % channels;
    data += c * height * width;
    int start_idx = group_map_range[group_id];
    int end_idx = group_map_range[group_id + 1];
    Dtype sumval = 0;
    for (int i = start_idx; i < end_idx; ++i) {
      sumval += data[group_map_index[i]];
    }
    group_mean_data[index] = sumval / (end_idx - start_idx) / group_channels;
  }
}

template <typename Dtype>
__global__ void GroupUnPoolForward(const int nthreads, const Dtype* group_mean_data,
    const int channels, const int height, const int width, const Dtype* group_data,
    const int num_groups, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int group_id = (int)group_data[h * width + w];
    top_data[index] += group_mean_data[c * num_groups + group_id];
  }
}

template <typename Dtype>
void GroupingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  // use the reordered internal group data
  const Dtype* group_data = group_blob_.gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
  const int dim = top[0]->count() / num_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  CHECK_EQ(bottom.size(), 2);
  for (int n = 0; n < num_; ++n) {
    const vector<map<int, vector<int> > >& group_maps = group_maps_vec_[n];
    for (int gc = 0; gc < group_channels_; ++gc) {
      const map<int, vector<int> >& group_map = group_maps[gc];
      const int num_groups = group_map.size();
      // cuda function cannot call STL function, convert it to Blob data
      GetMapData(group_map, &group_map_range_, &group_map_index_);
      const int* group_map_range = group_map_range_.gpu_data();
      const int* group_map_index = group_map_index_.gpu_data();
      // prepare group_mean_
      group_mean_.Reshape(1, channels_, 1, num_groups);
      Dtype* group_mean_data = group_mean_.mutable_gpu_data();
      int group_count = group_mean_.count();
      // compute group_mean_data
      ComputeGroupMean<Dtype><<<CAFFE_GET_BLOCKS(group_count), CAFFE_CUDA_NUM_THREADS>>>(
          group_count, bottom_data, channels_, height_, width_, num_groups,
          group_map_range, group_map_index, group_channels_, group_mean_data);
      // spread group_mean_data to top_data
      GroupUnPoolForward<Dtype><<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
          dim, group_mean_data, channels_, height_, width_,
          group_data, num_groups, top_data);
      group_data += bottom[1]->offset(0, 1);
    }
    bottom_data += bottom[0]->offset(1);
    top_data += top[0]->offset(1);
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void GroupUnPoolBackward(const int nthreads, const Dtype* group_mean_diff,
    const int channels, const int height, const int width, const Dtype* group_data,
    const int num_groups, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int group_id = (int)group_data[h * width + w];
    bottom_diff[index] += group_mean_diff[c * num_groups + group_id];
  }
}

template <typename Dtype>
void GroupingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  // use the reordered internal group data
  const Dtype* group_data = group_blob_.gpu_data();
  const int dim = bottom[0]->count() / top[0]->num();
  caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom_diff);
  // NOLINT_NEXT_LINE(whitespace/operators)
  CHECK_EQ(bottom.size(), 2);
  for (int n = 0; n < num_; ++n) {
    const vector<map<int, vector<int> > >& group_maps = group_maps_vec_[n];
    for (int gc = 0; gc < group_channels_; ++gc) {
      const map<int, vector<int> >& group_map = group_maps[gc];
      const int num_groups = group_map.size();
      // cuda function cannot call STL function, convert it to Blob data
      GetMapData(group_map, &group_map_range_, &group_map_index_);
      const int* group_map_range = group_map_range_.gpu_data();
      const int* group_map_index = group_map_index_.gpu_data();
      // prepare group_mean_
      group_mean_.Reshape(1, channels_, 1, num_groups);
      Dtype* group_mean_diff = group_mean_.mutable_gpu_diff();
      int group_count = group_mean_.count();
      // compute group_mean_diff
      ComputeGroupMean<Dtype><<<CAFFE_GET_BLOCKS(group_count), CAFFE_CUDA_NUM_THREADS>>>(
          group_count, top_diff, channels_, height_, width_, num_groups,
          group_map_range, group_map_index, group_channels_, group_mean_diff);
      // spread group_mean_diff to bottom_diff
      GroupUnPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
          dim, group_mean_diff, channels_, height_, width_, group_data,
          num_groups, bottom_diff);
      group_data += bottom[1]->offset(0, 1);
    }
    bottom_diff += bottom[0]->offset(1);
    top_diff += top[0]->offset(1);
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(GroupingLayer);


}  // namespace caffe
