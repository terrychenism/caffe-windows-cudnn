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
class GroupingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  GroupingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
      blob_top_(new Blob<Dtype>()) {}
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
  virtual ~GroupingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(GroupingLayerTest, TestDtypesAndDevices);

TYPED_TEST(GroupingLayerTest, TestSetupGroup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  GroupingLayer<Dtype> layer(layer_param);

  Blob<Dtype>* group_blob(new Blob<Dtype>());
  group_blob->Reshape(2, 1, 3, 2);
  this->blob_bottom_vec_.push_back(group_blob);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 2);
  this->blob_bottom_vec_.pop_back();
}

TYPED_TEST(GroupingLayerTest, TestForwardGroup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  // Input:
  //     [ 1 2 4 ]
  //     [ 2 3 2 ]
  //     [ 1 2 0 ]
  for (int i = 0; i < 9 * num * channels; i += 9) {
    this->blob_bottom_->mutable_cpu_data()[i+0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i+1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+2] = 4;
    this->blob_bottom_->mutable_cpu_data()[i+3] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+4] = 3;
    this->blob_bottom_->mutable_cpu_data()[i+5] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+6] = 1;
    this->blob_bottom_->mutable_cpu_data()[i+7] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+8] = 0;
  }
  Blob<Dtype>* group_blob(new Blob<Dtype>());
  group_blob->Reshape(num, 1, 3, 3);
  // Group:
  //     [ 1 1 2 ]
  //     [ 1 1 2 ]
  //     [ 4 4 4 ]
  for (int i = 0; i < 9 * num; i += 9) {
    group_blob->mutable_cpu_data()[i+0] = 1;
    group_blob->mutable_cpu_data()[i+1] = 1;
    group_blob->mutable_cpu_data()[i+2] = 2;
    group_blob->mutable_cpu_data()[i+3] = 1;
    group_blob->mutable_cpu_data()[i+4] = 1;
    group_blob->mutable_cpu_data()[i+5] = 2;
    group_blob->mutable_cpu_data()[i+6] = 4;
    group_blob->mutable_cpu_data()[i+7] = 4;
    group_blob->mutable_cpu_data()[i+8] = 4;
  }
  this->blob_bottom_vec_.push_back(group_blob);
  GroupingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-8;
  // Output:
  //     [ 2 2 3 ]
  //     [ 2 2 3 ]
  //     [ 1 1 1 ]
  for (int i = 0; i < 9 * num * channels; i += 9) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 1, epsilon);
  }
  this->blob_bottom_vec_.pop_back();
}

TYPED_TEST(GroupingLayerTest, TestForwardGroupMult) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  // Input:
  //     [ 1 2 4 ]
  //     [ 2 3 6 ]
  //     [ 2 2 5 ]
  for (int i = 0; i < 9 * num * channels; i += 9) {
    this->blob_bottom_->mutable_cpu_data()[i+0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i+1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+2] = 4;
    this->blob_bottom_->mutable_cpu_data()[i+3] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+4] = 3;
    this->blob_bottom_->mutable_cpu_data()[i+5] = 6;
    this->blob_bottom_->mutable_cpu_data()[i+6] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+7] = 2;
    this->blob_bottom_->mutable_cpu_data()[i+8] = 5;
  }
  Blob<Dtype>* group_blob(new Blob<Dtype>());
  group_blob->Reshape(num, 3, 3, 3);
  // Group:
  //     [ 1 1 2 ]
  //     [ 1 1 2 ]
  //     [ 4 4 4 ]
  //     =========
  //     [ 8 8 8 ]
  //     [ 8 8 8 ]
  //     [ 8 8 8 ]
  //     =========
  //     [ 1 1 1 ]
  //     [ 1 1 1 ]
  //     [ 2 2 2 ]
  for (int i = 0; i < 27 * num; i += 27) {
    group_blob->mutable_cpu_data()[i+0] = 1;
    group_blob->mutable_cpu_data()[i+1] = 1;
    group_blob->mutable_cpu_data()[i+2] = 2;
    group_blob->mutable_cpu_data()[i+3] = 1;
    group_blob->mutable_cpu_data()[i+4] = 1;
    group_blob->mutable_cpu_data()[i+5] = 2;
    group_blob->mutable_cpu_data()[i+6] = 4;
    group_blob->mutable_cpu_data()[i+7] = 4;
    group_blob->mutable_cpu_data()[i+8] = 4;
    group_blob->mutable_cpu_data()[i+9] = 8;
    group_blob->mutable_cpu_data()[i+10] = 8;
    group_blob->mutable_cpu_data()[i+11] = 8;
    group_blob->mutable_cpu_data()[i+12] = 8;
    group_blob->mutable_cpu_data()[i+13] = 8;
    group_blob->mutable_cpu_data()[i+14] = 8;
    group_blob->mutable_cpu_data()[i+15] = 8;
    group_blob->mutable_cpu_data()[i+16] = 8;
    group_blob->mutable_cpu_data()[i+17] = 8;
    group_blob->mutable_cpu_data()[i+18] = 1;
    group_blob->mutable_cpu_data()[i+19] = 1;
    group_blob->mutable_cpu_data()[i+20] = 1;
    group_blob->mutable_cpu_data()[i+21] = 1;
    group_blob->mutable_cpu_data()[i+22] = 1;
    group_blob->mutable_cpu_data()[i+23] = 1;
    group_blob->mutable_cpu_data()[i+24] = 2;
    group_blob->mutable_cpu_data()[i+25] = 2;
    group_blob->mutable_cpu_data()[i+26] = 2;
  }
  this->blob_bottom_vec_.push_back(group_blob);
  GroupingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-6;
  // Output:
  //     [ 8 8 11 ]
  //     [ 8 8 11 ]
  //     [ 9 9  9 ]
  for (int i = 0; i < 9 * num * channels; i += 9) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+0], 8.0/3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+1], 8.0/3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+2], 11.0/3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+3], 8.0/3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+4], 8.0/3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+5], 11.0/3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+6], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+7], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i+8], 3, epsilon);
  }
  this->blob_bottom_vec_.pop_back();
}

TYPED_TEST(GroupingLayerTest, TestGradientGroup) {
  typedef typename TypeParam::Dtype Dtype;
  Blob<Dtype>* group_blob(new Blob<Dtype>());
  int num = 2;
  group_blob->Reshape(num, 3, 3, 2);
  // Group:
  //     [ 1 2 ]
  //     [ 1 2 ]
  //     [ 4 4 ]
 //     =========
  //     [ 8 8 ]
  //     [ 8 8 ]
  //     [ 8 8 ]
  //     =========
  //     [ 1 1 ]
  //     [ 1 1 ]
  //     [ 2 2 ]
  for (int i = 0; i < 18 * num; i += 18) {
    group_blob->mutable_cpu_data()[i+0] = 1;
    group_blob->mutable_cpu_data()[i+1] = 2;
    group_blob->mutable_cpu_data()[i+2] = 1;
    group_blob->mutable_cpu_data()[i+3] = 2;
    group_blob->mutable_cpu_data()[i+4] = 4;
    group_blob->mutable_cpu_data()[i+5] = 4;
    group_blob->mutable_cpu_data()[i+6] = 8;
    group_blob->mutable_cpu_data()[i+7] = 8;
    group_blob->mutable_cpu_data()[i+8] = 8;
    group_blob->mutable_cpu_data()[i+9] = 8;
    group_blob->mutable_cpu_data()[i+10] = 8;
    group_blob->mutable_cpu_data()[i+11] = 8;
    group_blob->mutable_cpu_data()[i+12] = 1;
    group_blob->mutable_cpu_data()[i+13] = 1;
    group_blob->mutable_cpu_data()[i+14] = 1;
    group_blob->mutable_cpu_data()[i+15] = 1;
    group_blob->mutable_cpu_data()[i+16] = 2;
    group_blob->mutable_cpu_data()[i+17] = 2;
  }
  this->blob_bottom_vec_.push_back(group_blob);
  LayerParameter layer_param;
  GroupingLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_, 0);
  this->blob_bottom_vec_.pop_back();
}

}  // namespace caffe
