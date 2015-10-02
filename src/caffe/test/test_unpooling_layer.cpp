#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class UnPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  UnPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
      blob_top_(new Blob<Dtype>()),
      blob_top_mask_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1029);
    blob_bottom_->Reshape(2, 3, 3, 2);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~UnPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_top_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(UnPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(UnPoolingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 4);
}

TYPED_TEST(UnPoolingLayerTest, TestSetupStrided) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 7);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(UnPoolingLayerTest, TestSetupPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_out_pad(1);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_DIV);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(UnPoolingLayerTest, TestSetupAuto1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(0);
  unpooling_param->set_out_stride(0);
  UnPoolingLayer<Dtype> layer(layer_param);

  Blob<Dtype>* unpool_blob(new Blob<Dtype>());
  unpool_blob->Reshape(2, 3, 6, 4);
  this->blob_bottom_vec_.push_back(unpool_blob);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 4);
  this->blob_bottom_vec_.pop_back();
}

TYPED_TEST(UnPoolingLayerTest, TestSetupAuto2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(0);
  unpooling_param->set_out_stride(0);
  UnPoolingLayer<Dtype> layer(layer_param);

  Blob<Dtype>* unpool_blob(new Blob<Dtype>());
  unpool_blob->Reshape(2, 3, 7, 5);
  this->blob_bottom_vec_.push_back(unpool_blob);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 7);
  EXPECT_EQ(this->blob_top_->width(), 6);
  this->blob_bottom_vec_.pop_back();
}

TYPED_TEST(UnPoolingLayerTest, TestSetupAuto3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(100);
  unpooling_param->set_out_stride(100);
  UnPoolingLayer<Dtype> layer(layer_param);

  Blob<Dtype>* unpool_blob(new Blob<Dtype>());
  unpool_blob->Reshape(2, 3, 7, 5);
  this->blob_bottom_vec_.push_back(unpool_blob);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 7);
  EXPECT_EQ(this->blob_top_->width(), 6);
  this->blob_bottom_vec_.pop_back();
}

// Test for 2 x 2 square unpooling layer
TYPED_TEST(UnPoolingLayerTest, TestForwardSquareFixed) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(2);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 2);
  // Input: 2 x 2 channels of:
  //     [1 2]
  //     [9 4]
  //     [5 3]
  for (int i = 0; i < 6 * num * channels; i += 6) {
    this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i +  2] = 9;
    this->blob_bottom_->mutable_cpu_data()[i +  3] = 4;
    this->blob_bottom_->mutable_cpu_data()[i +  4] = 5;
    this->blob_bottom_->mutable_cpu_data()[i +  5] = 3;
  }
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 3);

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Expected output: 2 x 2 channels of:
  //     [1 2 0]
  //     [9 4 0]
  //     [5 3 0]
  //     [0 0 0]
  for (int i = 0; i < 12 * num * channels; i += 12) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 0], 1);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 1], 2);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 2], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 3], 9);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 4], 4);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 5], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 6], 5);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 7], 3);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 8], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 9], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 10], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 11], 0);
  }
}

// Test for 3 x 2 rectangular unpooling layer with out_kernel_h > out_kernel_w
TYPED_TEST(UnPoolingLayerTest, TestForwardHighFixed) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_h(3);
  unpooling_param->set_out_kernel_w(2);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  // Input: 2 x 2 channels of:
  //     [1 2 8]
  //     [9 4 6]
  //     [5 3 7]
  for (int i = 0; i < 9 * num * channels; i += 9) {
    this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i +  2] = 8;
    this->blob_bottom_->mutable_cpu_data()[i +  3] = 9;
    this->blob_bottom_->mutable_cpu_data()[i +  4] = 4;
    this->blob_bottom_->mutable_cpu_data()[i +  5] = 6;
    this->blob_bottom_->mutable_cpu_data()[i +  6] = 5;
    this->blob_bottom_->mutable_cpu_data()[i +  7] = 3;
    this->blob_bottom_->mutable_cpu_data()[i +  8] = 7;
  }
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 4);

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Expected output: 2 x 2 channels of:
  //     [0 0 0 0]
  //     [1 2 8 0]
  //     [9 4 6 0]
  //     [5 3 7 0]
  //     [0 0 0 0]
  for (int i = 0; i < 20 * num * channels; i += 20) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 0], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 1], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 2], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 3], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 4], 1);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 5], 2);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 6], 8);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 7], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 8], 9);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 9], 4);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 10], 6);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 11], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 12], 5);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 13], 3);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 14], 7);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 15], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 16], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 17], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 18], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 19], 0);
  }
}

// Test for 2 x 3 rectangular unpooling layer with out_kernel_w > out_kernel_h
TYPED_TEST(UnPoolingLayerTest, TestForwardWideFixed) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_h(2);
  unpooling_param->set_out_kernel_w(3);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  // Input: 2 x 2 channels of:
  //     [1 2 8]
  //     [9 4 6]
  //     [5 3 7]
  for (int i = 0; i < 9 * num * channels; i += 9) {
    this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i +  2] = 8;
    this->blob_bottom_->mutable_cpu_data()[i +  3] = 9;
    this->blob_bottom_->mutable_cpu_data()[i +  4] = 4;
    this->blob_bottom_->mutable_cpu_data()[i +  5] = 6;
    this->blob_bottom_->mutable_cpu_data()[i +  6] = 5;
    this->blob_bottom_->mutable_cpu_data()[i +  7] = 3;
    this->blob_bottom_->mutable_cpu_data()[i +  8] = 7;
  }
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Expected output: 2 x 2 channels of:
  //     [0 1 2 8 0]
  //     [0 9 4 6 0]
  //     [0 5 3 7 0]
  //     [0 0 0 0 0]
  for (int i = 0; i < 20 * num * channels; i += 20) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 0], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 1], 1);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 2], 2);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 3], 8);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 4], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 5], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 6], 9);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 7], 4);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 8], 6);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 9], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 10], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 11], 5);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 12], 3);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 13], 7);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 14], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 15], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 16], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 17], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 18], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 19], 0);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardFixedStrided) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  // Input:
  //     [ 1 2 4 ]
  //     [ 2 3 2 ]
  //     [ 4 2 1 ]
  for (int i = 0; i < 9 * num * channels; i += 9) {
    this->blob_bottom_->mutable_cpu_data()[i+0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i+1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+2] = 4;
    this->blob_bottom_->mutable_cpu_data()[i+3] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+4] = 3;
    this->blob_bottom_->mutable_cpu_data()[i+5] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+6] = 4;
    this->blob_bottom_->mutable_cpu_data()[i+7] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+8] = 1;
  }
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 7);
  EXPECT_EQ(this->blob_top_->width(), 7);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-8;
  // Output:
  //     [ 0 0 0 0 0 0 0 ]
  //     [ 0 1 0 2 0 4 0 ]
  //     [ 0 0 0 0 0 0 0 ]
  //     [ 0 2 0 3 0 2 0 ]
  //     [ 0 0 0 0 0 0 0 ]
  //     [ 0 4 0 2 0 1 0 ]
  //     [ 0 0 0 0 0 0 0 ]
  for (int i = 0; i < 49 * num * channels; i += 49) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+9], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+10], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+11], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+12], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+13], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+14], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+15], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+16], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+17], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+18], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+19], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+20], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+21], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+22], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+23], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+24], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+25], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+26], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+27], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+28], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+29], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+30], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+31], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+32], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+33], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+34], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+35], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+36], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+37], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+38], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+39], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+40], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+41], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+42], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+43], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+44], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+45], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+46], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+47], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+48], 0, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardFixedPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_out_pad(1);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  // Input:
  //     [ 1 2 4 ]
  //     [ 2 3 2 ]
  //     [ 4 2 1 ]
  for (int i = 0; i < 9 * num * channels; i += 9) {
    this->blob_bottom_->mutable_cpu_data()[i+0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i+1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+2] = 4;
    this->blob_bottom_->mutable_cpu_data()[i+3] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+4] = 3;
    this->blob_bottom_->mutable_cpu_data()[i+5] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+6] = 4;
    this->blob_bottom_->mutable_cpu_data()[i+7] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+8] = 1;
  }
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 5);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-8;
  // Output:
  //     [ 1 0 2 0 4 ]
  //     [ 0 0 0 0 0 ]
  //     [ 2 0 3 0 2 ]
  //     [ 0 0 0 0 0 ]
  //     [ 4 0 2 0 1 ]
  for (int i = 0; i < 25 * num * channels; i += 25) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+9], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+10], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+11], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+12], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+13], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+14], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+15], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+16], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+17], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+18], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+19], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+20], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+21], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+22], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+23], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+24], 1, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardDiv) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_DIV);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 2);
  //     [1 2]
  //     [9 4]
  //     [5 3]
  for (int i = 0; i < 6 * num * channels; i += 6) {
    this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i +  2] = 9;
    this->blob_bottom_->mutable_cpu_data()[i +  3] = 4;
    this->blob_bottom_->mutable_cpu_data()[i +  4] = 5;
    this->blob_bottom_->mutable_cpu_data()[i +  5] = 3;
  }
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 4);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-5;
  //     [1.0  1.5  1.5 2.0]
  //     [5.0  4.0  4.0 3.0]
  //     [5.0  4.0  4.0 3.0] / 9.0
  //     [7.0 5.25 5.25 3.5]
  //     [5.0  4.0  4.0 3.0]
  for (int i = 0; i < 20 * num * channels; i += 20) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 1.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 1.5/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 1.5/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 5.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 3.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 5.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+9], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+10], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+11], 3.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+12], 7.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+13], 5.25/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+14], 5.25/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+15], 3.5/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+16], 5.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+17], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+18], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+19], 3.0/9, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardDivConst) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_DIV);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(Dtype(2));
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 5);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-5;
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], 2.0/9, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardDivStrided) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_DIV);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 2);
  //     [1 2]
  //     [9 4]
  //     [5 3]
  for (int i = 0; i < 6 * num * channels; i += 6) {
    this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i +  2] = 9;
    this->blob_bottom_->mutable_cpu_data()[i +  3] = 4;
    this->blob_bottom_->mutable_cpu_data()[i +  4] = 5;
    this->blob_bottom_->mutable_cpu_data()[i +  5] = 3;
  }
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 7);
  EXPECT_EQ(this->blob_top_->width(), 5);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  //     [1.0 1.0  1.5  2.0 2.0]
  //     [1.0 1.0  1.5  2.0 2.0]
  //     [5.0 5.0  4.0  3.0 3.0]
  //     [9.0 9.0  6.5  4.0 4.0] / 9.0
  //     [7.0 7.0 5.25  3.5 3.5]
  //     [5.0 5.0  4.0  3.0 3.0]
  //     [5.0 5.0  4.0  3.0 3.0]
  Dtype epsilon = 1e-5;
  for (int i = 0; i < 35 * num * channels; i += 35) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 1.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 1.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 1.5/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 1.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 1.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 1.5/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+9], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+10], 5.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+11], 5.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+12], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+13], 3.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+14], 3.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+15], 9.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+16], 9.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+17], 6.5/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+18], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+19], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+20], 7.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+21], 7.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+22], 5.25/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+23], 3.5/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+24], 3.5/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+25], 5.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+26], 5.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+27], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+28], 3.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+29], 3.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+30], 5.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+31], 5.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+32], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+33], 3.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+34], 3.0/9, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardDivStridedConst) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_DIV);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(Dtype(2));
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 7);
  EXPECT_EQ(this->blob_top_->width(), 7);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-5;
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], 2.0/9, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardDivPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_out_pad(1);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_DIV);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 2);
  //     [1 2]
  //     [9 4]
  //     [5 3]
  for (int i = 0; i < 6 * num * channels; i += 6) {
    this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i +  2] = 9;
    this->blob_bottom_->mutable_cpu_data()[i +  3] = 4;
    this->blob_bottom_->mutable_cpu_data()[i +  4] = 5;
    this->blob_bottom_->mutable_cpu_data()[i +  5] = 3;
  }
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  //     [1.0  1.5  2.0]
  //     [5.0  4.0  3.0]
  //     [9.0  6.5  4.0]  / 9.0
  //     [7.0 5.25  3.5]
  //     [5.0  4.0  3.0]
  Dtype epsilon = 1e-5;
  for (int i = 0; i < 15 * num * channels; i += 15) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 1.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 1.5/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 5.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 3.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 9.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 6.5/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+9], 7.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+10], 5.25/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+11], 3.5/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+12], 5.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+13], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+14], 3.0/9, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardDivPaddedConst) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_out_pad(1);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_DIV);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(Dtype(2));
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 5);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-5;
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], 2.0/9, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardDivSquare) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(3);
  unpooling_param->set_out_pad(1);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_DIV);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  //     [1 2 8]
  //     [9 4 6]
  //     [5 3 0]
  for (int i = 0; i < 9 * num * channels; i += 9) {
    this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i +  2] = 8;
    this->blob_bottom_->mutable_cpu_data()[i +  3] = 9;
    this->blob_bottom_->mutable_cpu_data()[i +  4] = 4;
    this->blob_bottom_->mutable_cpu_data()[i +  5] = 6;
    this->blob_bottom_->mutable_cpu_data()[i +  6] = 5;
    this->blob_bottom_->mutable_cpu_data()[i +  7] = 3;
    this->blob_bottom_->mutable_cpu_data()[i +  8] = 0;
  }
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 7);
  EXPECT_EQ(this->blob_top_->width(), 7);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-5;
  //     [1 1 2 2 2 8 8]
  //     [1 1 2 2 2 8 8]
  //     [9 9 4 4 4 6 6]
  //     [9 9 4 4 4 6 6] / 9.0
  //     [9 9 4 4 4 6 6]
  //     [5 5 3 3 3 0 0]
  //     [5 5 3 3 3 0 0]
  for (int i = 0; i < 49 * num * channels; i += 49) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 1.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 1.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 8.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 8.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 1.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 1.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+9], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+10], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+11], 2.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+12], 8.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+13], 8.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+14], 9.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+15], 9.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+16], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+17], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+18], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+19], 6.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+20], 6.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+21], 9.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+22], 9.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+23], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+24], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+25], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+26], 6.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+27], 6.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+28], 9.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+29], 9.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+30], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+31], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+32], 4.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+33], 6.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+34], 6.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+35], 5.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+36], 5.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+37], 3.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+38], 3.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+39], 3.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+40], 0.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+41], 0.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+42], 5.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+43], 5.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+44], 3.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+45], 3.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+46], 3.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+47], 0.0/9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+48], 0.0/9, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardRep) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_REP);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 2);
  //     [1 2]
  //     [9 4]
  //     [5 3]
  for (int i = 0; i < 6 * num * channels; i += 6) {
    this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i +  2] = 9;
    this->blob_bottom_->mutable_cpu_data()[i +  3] = 4;
    this->blob_bottom_->mutable_cpu_data()[i +  4] = 5;
    this->blob_bottom_->mutable_cpu_data()[i +  5] = 3;
  }
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 4);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-5;
  //     [1.0 1.5 1.5 2.0]
  //     [5.0 4.0 4.0 3.0]
  //     [5.0 4.0 4.0 3.0]
  //     [7.0 5.25 5.25 3.5]
  //     [5.0 4.0 4.0 3.0]
  for (int i = 0; i < 20 * num * channels; i += 20) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 1.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 1.5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 1.5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 2.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 5.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 4.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 4.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 3.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 5.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+9], 4.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+10], 4.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+11], 3.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+12], 7.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+13], 5.25, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+14], 5.25, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+15], 3.5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+16], 5.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+17], 4.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+18], 4.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+19], 3.0, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardRepConst) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_REP);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(Dtype(2));
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 5);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-5;
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], 2.0, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardRepStrided) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_REP);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 2);
  //     [1 2]
  //     [9 4]
  //     [5 3]
  for (int i = 0; i < 6 * num * channels; i += 6) {
    this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i +  2] = 9;
    this->blob_bottom_->mutable_cpu_data()[i +  3] = 4;
    this->blob_bottom_->mutable_cpu_data()[i +  4] = 5;
    this->blob_bottom_->mutable_cpu_data()[i +  5] = 3;
  }
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 7);
  EXPECT_EQ(this->blob_top_->width(), 5);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  //     [1.0 1.0 1.5 2.0 2.0]
  //     [1.0 1.0 1.5 2.0 2.0]
  //     [5.0 5.0 4.0 3.0 3.0]
  //     [9.0 9.0 6.5 4.0 4.0]
  //     [7.0 7.0 5.25 3.5 3.5]
  //     [5.0 5.0 4.0 3.0 3.0]
  //     [5.0 5.0 4.0 3.0 3.0]
  Dtype epsilon = 1e-5;
  for (int i = 0; i < 35 * num * channels; i += 35) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 1.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 1.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 1.5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 2.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 2.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 1.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 1.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 1.5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 2.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+9], 2.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+10], 5.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+11], 5.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+12], 4.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+13], 3.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+14], 3.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+15], 9.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+16], 9.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+17], 6.5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+18], 4.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+19], 4.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+20], 7.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+21], 7.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+22], 5.25, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+23], 3.5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+24], 3.5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+25], 5.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+26], 5.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+27], 4.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+28], 3.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+29], 3.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+30], 5.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+31], 5.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+32], 4.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+33], 3.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+34], 3.0, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardRepStridedConst) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_REP);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(Dtype(2));
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 7);
  EXPECT_EQ(this->blob_top_->width(), 7);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-5;
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], 2.0, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardRepPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_out_pad(1);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_REP);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 2);
  //     [1 2]
  //     [9 4]
  //     [5 3]
  for (int i = 0; i < 6 * num * channels; i += 6) {
    this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i +  2] = 9;
    this->blob_bottom_->mutable_cpu_data()[i +  3] = 4;
    this->blob_bottom_->mutable_cpu_data()[i +  4] = 5;
    this->blob_bottom_->mutable_cpu_data()[i +  5] = 3;
  }
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  //     [1.0 1.5 2.0]
  //     [5.0 4.0 3.0]
  //     [9.0 6.5 4.0]
  //     [7.0 5.25 3.5]
  //     [5.0 4.0 3.0]
  Dtype epsilon = 1e-5;
  for (int i = 0; i < 15 * num * channels; i += 15) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 1.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 1.5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 2.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 5.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 4.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 3.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 9.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 6.5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 4.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+9], 7.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+10], 5.25, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+11], 3.5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+12], 5.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+13], 4.0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+14], 3.0, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardRepPaddedConst) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_out_pad(1);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_REP);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(Dtype(2));
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 5);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-5;
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], 2.0, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForwardRepSquare) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(3);
  unpooling_param->set_out_pad(1);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_REP);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  //     [1 2 8]
  //     [9 4 6]
  //     [5 3 0]
  for (int i = 0; i < 9 * num * channels; i += 9) {
    this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i +  2] = 8;
    this->blob_bottom_->mutable_cpu_data()[i +  3] = 9;
    this->blob_bottom_->mutable_cpu_data()[i +  4] = 4;
    this->blob_bottom_->mutable_cpu_data()[i +  5] = 6;
    this->blob_bottom_->mutable_cpu_data()[i +  6] = 5;
    this->blob_bottom_->mutable_cpu_data()[i +  7] = 3;
    this->blob_bottom_->mutable_cpu_data()[i +  8] = 0;
  }
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 7);
  EXPECT_EQ(this->blob_top_->width(), 7);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-5;
  //     [1 1 2 2 2 8 8]
  //     [1 1 2 2 2 8 8]
  //     [9 9 4 4 4 6 6]
  //     [9 9 4 4 4 6 6]
  //     [9 9 4 4 4 6 6]
  //     [5 5 3 3 3 0 0]
  //     [5 5 3 3 3 0 0]
  for (int i = 0; i < 49 * num * channels; i += 49) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 8, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 8, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+9], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+10], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+11], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+12], 8, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+13], 8, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+14], 9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+15], 9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+16], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+17], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+18], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+19], 6, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+20], 6, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+21], 9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+22], 9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+23], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+24], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+25], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+26], 6, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+27], 6, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+28], 9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+29], 9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+30], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+31], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+32], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+33], 6, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+34], 6, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+35], 5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+36], 5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+37], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+38], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+39], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+40], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+41], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+42], 5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+43], 5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+44], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+45], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+46], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+47], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+48], 0, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestGradientFixed) {
  typedef typename TypeParam::Dtype Dtype;
  for (int out_kernel_h = 3; out_kernel_h <= 4; out_kernel_h++) {
    for (int out_kernel_w = 3; out_kernel_w <= 4; out_kernel_w++) {
      LayerParameter layer_param;
      UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
      unpooling_param->set_out_kernel_h(out_kernel_h);
      unpooling_param->set_out_kernel_w(out_kernel_w);
      unpooling_param->set_out_stride(2);
      unpooling_param->set_out_pad(1);
      unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
      UnPoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                      this->blob_top_vec_);
    }
  }
}

TYPED_TEST(UnPoolingLayerTest, TestGradientDiv) {
  typedef typename TypeParam::Dtype Dtype;
  for (int out_kernel_h = 3; out_kernel_h <= 4; out_kernel_h++) {
    for (int out_kernel_w = 3; out_kernel_w <= 4; out_kernel_w++) {
      LayerParameter layer_param;
      UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
      unpooling_param->set_out_kernel_h(out_kernel_h);
      unpooling_param->set_out_kernel_w(out_kernel_w);
      unpooling_param->set_out_stride(2);
      unpooling_param->set_out_pad(1);
      unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_DIV);
      UnPoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                      this->blob_top_vec_);
    }
  }
}

TYPED_TEST(UnPoolingLayerTest, TestGradientRep) {
  typedef typename TypeParam::Dtype Dtype;
  for (int out_kernel_h = 3; out_kernel_h <= 4; out_kernel_h++) {
    for (int out_kernel_w = 3; out_kernel_w <= 4; out_kernel_w++) {
      LayerParameter layer_param;
      UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
      unpooling_param->set_out_kernel_h(out_kernel_h);
      unpooling_param->set_out_kernel_w(out_kernel_w);
      unpooling_param->set_out_stride(2);
      unpooling_param->set_out_pad(1);
      unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_REP);
      UnPoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                      this->blob_top_vec_);
    }
  }
}

}  // namespace caffe
