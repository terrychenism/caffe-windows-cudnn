#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

// Forward declare kernel functions
template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels,
    const int patch_h, const int patch_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int filter_stride_h,
    const int filter_stride_w, const int height_col, const int width_col,
    Dtype* data_im);

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class Col2ImKernelTest : public ::testing::Test {
 protected:
  Col2ImKernelTest()
        // big so launches > 1024 threads
      : blob_bottom_(new Blob<Dtype>(5, 500, 10, 10)),
        blob_top_(new Blob<Dtype>()),
        blob_top_cpu_(new Blob<Dtype>()) {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    height_col_ = blob_bottom_->height();
    width_col_ = blob_bottom_->width();
    channels_col_ = blob_bottom_->channels();
    pad_ = 1;
    stride_ = 1;
    filter_stride_ = 2;
    kernel_size_ = 2;
    channels_ = channels_col_ / kernel_size_ / kernel_size_;
    const int kernel_size_eff = kernel_size_
      + (kernel_size_ - 1) * (filter_stride_ - 1);
    height_ = (height_col_ - 1) * stride_ + kernel_size_eff - 2 * pad_;
    width_ = (width_col_ - 1) * stride_ + kernel_size_eff - 2 * pad_;
  }

  virtual ~Col2ImKernelTest() {
      delete blob_bottom_;
      delete blob_top_;
      delete blob_top_cpu_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_cpu_;
  int height_;
  int width_;
  int channels_;
  int pad_;
  int stride_;
  int filter_stride_;
  int kernel_size_;
  int height_col_;
  int width_col_;
  int channels_col_;
};

TYPED_TEST_CASE(Col2ImKernelTest, TestDtypes);

TYPED_TEST(Col2ImKernelTest, TestGPU) {
  Caffe::set_mode(Caffe::GPU);

  // Reshape the blobs to correct size for Col2Im output
  this->blob_top_->Reshape(this->blob_bottom_->num(),
          this->channels_, this->height_, this->width_);

  this->blob_top_cpu_->Reshape(this->blob_bottom_->num(),
          this->channels_, this->height_, this->width_);

  const TypeParam* bottom_data = this->blob_bottom_->gpu_data();
  TypeParam* top_data = this->blob_top_->mutable_gpu_data();
  TypeParam* cpu_data = this->blob_top_cpu_->mutable_cpu_data();

  // CPU Version
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    col2im_cpu(this->blob_bottom_->cpu_data() + this->blob_bottom_->offset(n),
      this->channels_, this->height_, this->width_,
      this->kernel_size_, this->kernel_size_, this->pad_, this->pad_,
      this->stride_, this->stride_, this->filter_stride_, this->filter_stride_,
      cpu_data + this->blob_top_cpu_->offset(n));
  }

  // GPU version
  int num_kernels = this->channels_ * this->height_ * this->width_;
  int default_grid_dim = CAFFE_GET_BLOCKS(num_kernels);

  // Launch with different grid sizes
  for (int grid_div = 2; grid_div <= 8; grid_div++) {
    for (int n = 0; n < this->blob_bottom_->num(); ++n) {
      int grid_dim = default_grid_dim/grid_div;
      // NOLINT_NEXT_LINE(whitespace/operators)
      col2im_gpu_kernel<TypeParam><<<grid_dim, CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, bottom_data + this->blob_bottom_->offset(n),
        this->height_, this->width_, this->channels_, this->kernel_size_,
        this->kernel_size_, this->pad_, this->pad_, this->stride_,
        this->stride_, this->filter_stride_, this->filter_stride_,
        this->height_col_, this->width_col_,
        top_data + this->blob_top_->offset(n));
      CUDA_POST_KERNEL_CHECK;
    }

    // Compare results against CPU version
    for (int i = 0; i < this->blob_top_->count(); ++i) {
      TypeParam cpuval = cpu_data[i];
      TypeParam gpuval = this->blob_top_->cpu_data()[i];
      EXPECT_NEAR(cpuval, gpuval, 1e-5);
    }
  }
}

}  // namespace caffe
