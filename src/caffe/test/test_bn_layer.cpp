#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define BATCH_SIZE 5
#define INPUT_DATA_SIZE 3

namespace caffe {

template <typename TypeParam>
class BNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  BNLayerTest()
      : blob_bottom_(new Blob<Dtype>(8, 2, 3, 4)),
      blob_top_(new Blob<Dtype>()) {
        // fill the values
        FillerParameter filler_param;
        GaussianFiller<Dtype> filler(filler_param);
        filler.Fill(this->blob_bottom_);
        blob_bottom_vec_.push_back(blob_bottom_);
        blob_top_vec_.push_back(blob_top_);
      }
  virtual ~BNLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BNLayerTest, TestDtypesAndDevices);

TYPED_TEST(BNLayerTest, TestForwardScaleOneShiftZero) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_bn_param()->mutable_scale_filler()->set_value(1);
  layer_param.mutable_bn_param()->mutable_shift_filler()->set_value(0);

  BNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int j = 0; j < channels; ++j) {
    Dtype sum = 0, var = 0;
    for (int i = 0; i < num; ++i) {
      for ( int k = 0; k < height; ++k ) {
        for ( int l = 0; l < width; ++l ) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
      }
    }
    sum /= height * width * num;
    var /= height * width * num;

    const Dtype kErrorBound = 0.001;
    // expect zero mean
    EXPECT_NEAR(0, sum, kErrorBound);
    // expect unit variance
    EXPECT_NEAR(1, var, kErrorBound);
  }
}

TYPED_TEST(BNLayerTest, TestForwardScaleOneShiftZeroEltwise) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_bn_param()->mutable_scale_filler()->set_value(1);
  layer_param.mutable_bn_param()->mutable_shift_filler()->set_value(0);
  layer_param.mutable_bn_param()->set_across_spatial(false);

  BNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int j = 0; j < channels; ++j) {
    for ( int k = 0; k < height; ++k ) {
      for ( int l = 0; l < width; ++l ) {
        Dtype sum = 0, var = 0;
        for (int i = 0; i < num; ++i) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
        sum /= num;
        var /= num;
        const Dtype kErrorBound = 0.001;
        // expect zero mean
        EXPECT_NEAR(0, sum, kErrorBound);
        // expect unit variance
        EXPECT_NEAR(1, var, kErrorBound);
      }
    }
  }
}

TYPED_TEST(BNLayerTest, TestForwardScaleOneShiftOne) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_bn_param()->mutable_scale_filler()->set_value(1);
  layer_param.mutable_bn_param()->mutable_shift_filler()->set_value(1);
  BNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int j = 0; j < channels; ++j) {
    Dtype sum = 0, var = 0;
    for (int i = 0; i < num; ++i) {
      for ( int k = 0; k < height; ++k ) {
        for ( int l = 0; l < width; ++l ) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
      }
    }
    sum /= height * width * num;
    var /= height * width * num;

    const Dtype kErrorBound = 0.001;
    // expect zero mean
    EXPECT_NEAR(1, sum, kErrorBound);
    // expect unit variance
    EXPECT_NEAR(1+1, var, kErrorBound);
  }
}

TYPED_TEST(BNLayerTest, TestForwardScaleOneShiftOneEltwise) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_bn_param()->mutable_scale_filler()->set_value(1);
  layer_param.mutable_bn_param()->mutable_shift_filler()->set_value(1);
  layer_param.mutable_bn_param()->set_across_spatial(false);
  BNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int j = 0; j < channels; ++j) {
    for ( int k = 0; k < height; ++k ) {
      for ( int l = 0; l < width; ++l ) {
        Dtype sum = 0, var = 0;
        for (int i = 0; i < num; ++i) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
        sum /= num;
        var /= num;

        const Dtype kErrorBound = 0.001;
        // expect zero mean
        EXPECT_NEAR(1, sum, kErrorBound);
        // expect unit variance
        EXPECT_NEAR(1+1, var, kErrorBound);
      }
    }
  }
}

TYPED_TEST(BNLayerTest, TestForwardScaleTwoShiftOne) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_bn_param()->mutable_scale_filler()->set_value(2);
  layer_param.mutable_bn_param()->mutable_shift_filler()->set_value(1);

  BNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int j = 0; j < channels; ++j) {
    Dtype sum = 0, var = 0;
    for (int i = 0; i < num; ++i) {
      for ( int k = 0; k < height; ++k ) {
        for ( int l = 0; l < width; ++l ) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
      }
    }
    sum /= height * width * num;
    var /= height * width * num;

    const Dtype kErrorBound = 0.001;
    // expect zero mean
    EXPECT_NEAR(1, sum, kErrorBound);
    // expect unit variance
    EXPECT_NEAR(4+1, var, kErrorBound);
  }
}

TYPED_TEST(BNLayerTest, TestForwardScaleTwoShiftOneEltwise) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_bn_param()->mutable_scale_filler()->set_value(2);
  layer_param.mutable_bn_param()->mutable_shift_filler()->set_value(1);
  layer_param.mutable_bn_param()->set_across_spatial(false);

  BNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int j = 0; j < channels; ++j) {
    for ( int k = 0; k < height; ++k ) {
      for ( int l = 0; l < width; ++l ) {
        Dtype sum = 0, var = 0;
        for (int i = 0; i < num; ++i) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
        sum /= num;
        var /= num;

        const Dtype kErrorBound = 0.001;
        // expect zero mean
        EXPECT_NEAR(1, sum, kErrorBound);
        // expect unit variance
        EXPECT_NEAR(4+1, var, kErrorBound);
      }
    }
  }
}

TYPED_TEST(BNLayerTest, TestGradientShiftZero) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_bn_param()->mutable_scale_filler()->set_value(1);
  layer_param.mutable_bn_param()->mutable_shift_filler()->set_value(0);

  BNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

TYPED_TEST(BNLayerTest, TestGradientShiftZeroEltwise) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_bn_param()->mutable_scale_filler()->set_value(1);
  layer_param.mutable_bn_param()->mutable_shift_filler()->set_value(0);
  layer_param.mutable_bn_param()->set_across_spatial(false);

  BNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-3, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

TYPED_TEST(BNLayerTest, TestGradientShiftOne) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_bn_param()->mutable_scale_filler()->set_value(1);
  layer_param.mutable_bn_param()->mutable_shift_filler()->set_value(1);

  BNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

TYPED_TEST(BNLayerTest, TestGradientShiftOneEltwise) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_bn_param()->mutable_scale_filler()->set_value(1);
  layer_param.mutable_bn_param()->mutable_shift_filler()->set_value(1);
  layer_param.mutable_bn_param()->set_across_spatial(false);

  BNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-3, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

TYPED_TEST(BNLayerTest, TestGradientScaleTwoShiftOne) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_bn_param()->mutable_scale_filler()->set_value(2);
  layer_param.mutable_bn_param()->mutable_shift_filler()->set_value(1);

  BNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

TYPED_TEST(BNLayerTest, TestGradientScaleTwoShiftOneEltwise) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_bn_param()->mutable_scale_filler()->set_value(2);
  layer_param.mutable_bn_param()->mutable_shift_filler()->set_value(1);
  layer_param.mutable_bn_param()->set_across_spatial(false);

  BNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-3, 2e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

}  // namespace caffe
