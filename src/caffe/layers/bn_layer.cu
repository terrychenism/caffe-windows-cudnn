#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* const_top_data = top[0]->gpu_data();
  const int batch_count = bottom[0]->count();

  Dtype* mean_data = batch_mean_.mutable_gpu_data();
  const Dtype* const_mean_data = batch_mean_.gpu_data();
  Dtype* std_data = batch_std_.mutable_gpu_data();
  const Dtype* const_std_data = batch_std_.gpu_data();
  const int mean_count = batch_mean_.count();

  Dtype* accum_mean = accum_mean_.mutable_gpu_data();
  const Dtype* const_accum_mean = accum_mean_.gpu_data();
  Dtype* accum_variance = accum_variance_.mutable_gpu_data();
  const Dtype* const_accum_variance = accum_variance_.gpu_data();

  const Dtype* batch_sum_multiplier_data = batch_sum_multiplier_.gpu_data();
  const Dtype* spatial_sum_multiplier_data = NULL;
  if (across_spatial_) {
    spatial_sum_multiplier_data = spatial_sum_multiplier_.gpu_data();
  }
  Dtype* x_norm_data = x_norm_.mutable_gpu_data();

  const Dtype* scale_data = this->blobs_[0]->gpu_data();
  const Dtype* shift_data = this->blobs_[1]->gpu_data();

  Dtype* buffer_data = buffer_blob_.mutable_gpu_data();
  const Dtype* const_buffer_data = buffer_blob_.gpu_data();
  Dtype* buffer_cube = buffer_cube_.mutable_gpu_data();
  const Dtype* const_buffer_cube = buffer_cube_.gpu_data();

  // determine if we need to compute mean & std or use previous stored ones.
  if (this->phase_ == TEST && moving_average_) {
    // using moving average mean & variance
    caffe_copy(mean_count, const_accum_mean, mean_data);
    caffe_copy(mean_count, const_accum_variance, std_data);
  } else {
    // compute E(X)
    if (across_spatial_) {
      // average across batch
      caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_*H_*W_, Dtype(1./N_), bottom_data,
          batch_sum_multiplier_data, Dtype(0), buffer_cube);
      // average spatially
      caffe_gpu_gemv<Dtype>(CblasNoTrans, C_, H_*W_, Dtype(1./(H_*W_)),
          const_buffer_cube, spatial_sum_multiplier_data,
          Dtype(0), mean_data);
    } else {
      // average across batch
      caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_*H_*W_, Dtype(1./N_), bottom_data,
          batch_sum_multiplier_data, Dtype(0), mean_data);
    }

    // compute E(X^2)
    // put the square of bottom into buffer
    caffe_gpu_powx(batch_count, bottom_data, Dtype(2), buffer_data);
    if (across_spatial_) {
      // average across batch
      caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_*H_*W_, Dtype(1./N_), const_buffer_data,
          batch_sum_multiplier_data, Dtype(0), buffer_cube);
      // average spatially
      caffe_gpu_gemv<Dtype>(CblasNoTrans, C_, H_*W_, Dtype(1./(H_*W_)), const_buffer_cube,
          spatial_sum_multiplier_data, Dtype(0), std_data);
    } else {
      // average across batch
      caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_*H_*W_, Dtype(1./N_), const_buffer_data,
          batch_sum_multiplier_data, Dtype(0), std_data);
    }

    // compute (EX)^2, and temporary use part of buffer_data to store it
    caffe_gpu_powx(mean_count, const_mean_data, Dtype(2), buffer_cube);

    // compute variance using var(X) = E(X^2) - (EX)^2
    caffe_gpu_sub(mean_count, const_std_data, const_buffer_cube, std_data);

    if (this->phase_ == TRAIN && moving_average_) {
      // aggregate the mean & variance
      caffe_gpu_axpby(mean_count, decay_, const_mean_data, Dtype(1) - decay_,
                      accum_mean);
      caffe_gpu_axpby(mean_count, decay_, const_std_data, Dtype(1) - decay_,
                      accum_variance);
    }
  }

  // zero-mean & unit variance normalization
  // replicate mean
  if (across_spatial_) {
    // replicate spatially
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, H_*W_, 1, Dtype(1),
                          const_mean_data, spatial_sum_multiplier_data,
                          Dtype(0), buffer_cube);
    // replicate batch
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_*H_*W_, 1, Dtype(1),
                          batch_sum_multiplier_data, const_buffer_cube, Dtype(0),
                          buffer_data);
  } else {
    // replicate batch
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_*H_*W_, 1, Dtype(1),
                          batch_sum_multiplier_data, const_mean_data, Dtype(0),
                          buffer_data);
  }
  // subtract mean
  caffe_gpu_sub(batch_count, bottom_data, const_buffer_data, top_data);

  // add epsilon to variance
  caffe_gpu_add_scalar(mean_count, var_eps_, std_data);
  // compute std
  caffe_gpu_powx(mean_count, const_std_data, Dtype(0.5), std_data);

  // replicate std
  if (across_spatial_) {
    // replicate spatially
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, H_*W_, 1, Dtype(1),
                          const_std_data, spatial_sum_multiplier_data, Dtype(0),
                          buffer_cube);
    // replicate batch
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_*H_*W_, 1, Dtype(1),
                          batch_sum_multiplier_data, const_buffer_cube,
                          Dtype(0), buffer_data);
  } else {
    // replicate batch
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_*H_*W_, 1, Dtype(1),
                          batch_sum_multiplier_data, const_std_data, Dtype(0),
                          buffer_data);
  }
  // divide std
  caffe_gpu_div(batch_count, const_top_data, const_buffer_data, top_data);

  // save x_norm
  caffe_copy(batch_count, const_top_data, x_norm_data);

  // scale
  if (across_spatial_) {
    // replicate spatially
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, H_*W_, 1, Dtype(1),
                          scale_data, spatial_sum_multiplier_data, Dtype(0),
                          buffer_cube);
    // replicate batch
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_*H_*W_, 1, Dtype(1),
                          batch_sum_multiplier_data, const_buffer_cube,
                          Dtype(0), buffer_data);
  } else {
    // replicate batch
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_*H_*W_, 1, Dtype(1),
                          batch_sum_multiplier_data, scale_data, Dtype(0),
                          buffer_data);
  }
  caffe_gpu_mul(batch_count, const_top_data, const_buffer_data, top_data);

  // shift
  if (across_spatial_) {
    // replicate spatially
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, H_*W_, 1, Dtype(1),
                          shift_data, spatial_sum_multiplier_data, Dtype(0),
                          buffer_cube);
    // replicate batch
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_*H_*W_, 1, Dtype(1),
                          batch_sum_multiplier_data, const_buffer_cube,
                          Dtype(0), buffer_data);
  } else {
    // replicate batch
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_*H_*W_, 1, Dtype(1),
                          batch_sum_multiplier_data, shift_data, Dtype(0),
                          buffer_data);
  }
  caffe_gpu_add(batch_count, const_top_data, const_buffer_data, top_data);
}

template <typename Dtype>
void BNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down,
                                  const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* std_data = batch_std_.gpu_data();
  const Dtype* x_norm_data = x_norm_.gpu_data();
  const int batch_count = top[0]->count();

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* const_bottom_diff = bottom[0]->gpu_diff();
  Dtype* scale_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* shift_diff = this->blobs_[1]->mutable_gpu_diff();
  const Dtype* scale_data = this->blobs_[0]->gpu_data();

  const Dtype* batch_sum_multiplier_data = batch_sum_multiplier_.gpu_data();
  const Dtype* spatial_sum_multiplier_data = NULL;
  if (across_spatial_) {
    spatial_sum_multiplier_data = spatial_sum_multiplier_.gpu_data();
  }

  Dtype* buffer_data = buffer_blob_.mutable_gpu_data();
  const Dtype* const_buffer_data = buffer_blob_.gpu_data();
  Dtype* buffer_cube = buffer_cube_.mutable_gpu_data();
  const Dtype* const_buffer_cube = buffer_cube_.gpu_data();
  Dtype* buffer_vec = buffer_vec_.mutable_gpu_data();
  const Dtype* const_buffer_vec = buffer_vec_.gpu_data();

  // Propagate layer to parameters
  // gradient w.r.t. scale
  caffe_gpu_mul(batch_count, x_norm_data, top_diff, buffer_data);
  if (across_spatial_) {
    // sum across batch
    caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_*H_*W_, Dtype(1), const_buffer_data,
                          batch_sum_multiplier_data, Dtype(0), buffer_cube);
    // sum spatially
    caffe_gpu_gemv<Dtype>(CblasNoTrans, C_, H_*W_, Dtype(1), const_buffer_cube,
                          spatial_sum_multiplier_data, Dtype(0), scale_diff);
  } else {
    // sum across batch
    caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_*H_*W_, Dtype(1), const_buffer_data,
                          batch_sum_multiplier_data, Dtype(0), scale_diff);
  }

  // gradient w.r.t. shift
  if (across_spatial_) {
    // sum across batch
    caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_*H_*W_, Dtype(1), top_diff, 
                          batch_sum_multiplier_data, Dtype(0), buffer_cube);
    // sum spatially
    caffe_gpu_gemv<Dtype>(CblasNoTrans, C_, H_*W_, Dtype(1), const_buffer_cube,
                          spatial_sum_multiplier_data, Dtype(0), shift_diff);
  } else {
    // sum across batch
    caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_*H_*W_, Dtype(1), top_diff,
                          batch_sum_multiplier_data, Dtype(0), shift_diff);
  }

  // propagate top_diff to bottom_diff
  // compute derivative over x_norm
  if (across_spatial_) {
    // replicate spatially
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, H_*W_, 1, Dtype(1),
                          scale_data, spatial_sum_multiplier_data, Dtype(0),
                          buffer_cube);
    // replicate batch
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_*H_*W_, 1, Dtype(1),
                          batch_sum_multiplier_data, const_buffer_cube,
                          Dtype(0), buffer_data);
  } else {
    // replicate batch
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_*H_*W_, 1, Dtype(1),
                          batch_sum_multiplier_data, scale_data, Dtype(0),
                          buffer_data);
  }
  caffe_gpu_mul(batch_count, top_diff, const_buffer_data, buffer_data);

  // compute 1
  caffe_gpu_mul(batch_count, x_norm_data, const_buffer_data, bottom_diff);
  // sum across batch
  caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_*H_*W_, Dtype(1), const_bottom_diff,
                        batch_sum_multiplier_data, Dtype(0), buffer_cube);
  if (across_spatial_) {
    // sum spatially
    caffe_gpu_gemv<Dtype>(CblasNoTrans, C_, H_*W_, Dtype(1), const_buffer_cube,
                          spatial_sum_multiplier_data, Dtype(0), buffer_vec);
    // replicate spatially
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, H_*W_, 1, Dtype(1),
                          const_buffer_vec, spatial_sum_multiplier_data, Dtype(0),
                          buffer_cube);
  }
  // replicate batch
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_*H_*W_, 1, Dtype(1),
                        batch_sum_multiplier_data, const_buffer_cube, Dtype(0),
                        bottom_diff);

  // compute 2
  caffe_gpu_mul(batch_count, x_norm_data, const_bottom_diff, bottom_diff);

  // compute 3
  // sum across batch
  caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_*H_*W_, Dtype(1), const_buffer_data,
                        batch_sum_multiplier_data, Dtype(0), buffer_cube);
  if (across_spatial_) {
    // sum spatially
    caffe_gpu_gemv<Dtype>(CblasNoTrans, C_, H_*W_, Dtype(1), const_buffer_cube,
                          spatial_sum_multiplier_data, Dtype(0), buffer_vec);
    // replicate spatially
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, H_*W_, 1, Dtype(1),
                          const_buffer_vec, spatial_sum_multiplier_data, Dtype(0),
                          buffer_cube);
  }
  // replicate batch and add it to bottom_diff
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_*H_*W_, 1, Dtype(1),
                        batch_sum_multiplier_data, const_buffer_cube, Dtype(1),
                        bottom_diff);

  // compute 4
  caffe_gpu_axpby(batch_count, Dtype(1), const_buffer_data,
                  Dtype(-BH_*BW_)/(N_*H_*W_), bottom_diff);

  // compute 5 by dividing std
  if (across_spatial_) {
    // replicate spatially
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, H_*W_, 1, Dtype(1),
                          std_data, spatial_sum_multiplier_data, Dtype(0),
                          buffer_cube);
    // replicate batch
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_*H_*W_, 1, Dtype(1),
                          batch_sum_multiplier_data, const_buffer_cube,
                          Dtype(0), buffer_data);
  } else {
    // replicate batch
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_*H_*W_, 1, Dtype(1),
                          batch_sum_multiplier_data, std_data, Dtype(0),
                          buffer_data);
  }
  caffe_gpu_div(batch_count, const_bottom_diff, const_buffer_data, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(BNLayer);
}  // namespace caffe
