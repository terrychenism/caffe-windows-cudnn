#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class ParseEvaluateLayerTest : public ::testing::Test {
 protected:
  ParseEvaluateLayerTest()
      : blob_bottom_prediction_(new Blob<Dtype>()),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        num_labels_(5) {
    vector<int> shape(4);
    shape[0] = 2;
    shape[1] = 1;
    shape[2] = 3;
    shape[3] = 3;
    blob_bottom_prediction_->Reshape(shape);
    blob_bottom_label_->Reshape(shape);
    FillBottoms();

    shape[0] = 1;
    shape[1] = num_labels_;
    shape[2] = 1;
    shape[3] = 3;
    blob_top_->Reshape(shape);

    blob_bottom_vec_.push_back(blob_bottom_prediction_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ParseEvaluateLayerTest() {
    delete blob_bottom_prediction_;
    delete blob_bottom_label_;
    delete blob_top_;
  }

  void FillBottoms() {
    // manually fill some data
    // Input: (prediction)
    //     [ 1 1 2 ]
    //     [ 1 1 2 ]
    //     [ 1 3 4 ]
    //     =========
    //     [ 0 3 2 ]
    //     [ 3 3 2 ]
    //     [ 2 0 0 ]
    blob_bottom_prediction_->mutable_cpu_data()[0] = 1;
    blob_bottom_prediction_->mutable_cpu_data()[1] = 1;
    blob_bottom_prediction_->mutable_cpu_data()[2] = 2;
    blob_bottom_prediction_->mutable_cpu_data()[3] = 1;
    blob_bottom_prediction_->mutable_cpu_data()[4] = 1;
    blob_bottom_prediction_->mutable_cpu_data()[5] = 2;
    blob_bottom_prediction_->mutable_cpu_data()[6] = 1;
    blob_bottom_prediction_->mutable_cpu_data()[7] = 3;
    blob_bottom_prediction_->mutable_cpu_data()[8] = 4;
    blob_bottom_prediction_->mutable_cpu_data()[9] = 0;
    blob_bottom_prediction_->mutable_cpu_data()[10] = 3;
    blob_bottom_prediction_->mutable_cpu_data()[11] = 2;
    blob_bottom_prediction_->mutable_cpu_data()[12] = 3;
    blob_bottom_prediction_->mutable_cpu_data()[13] = 3;
    blob_bottom_prediction_->mutable_cpu_data()[14] = 2;
    blob_bottom_prediction_->mutable_cpu_data()[15] = 2;
    blob_bottom_prediction_->mutable_cpu_data()[16] = 0;
    blob_bottom_prediction_->mutable_cpu_data()[17] = 0;
    // Input: (label)
    //     [ 1 1 0 ]
    //     [ 1 0 2 ]
    //     [ 2 2 0 ]
    //     =========
    //     [ 1 2 2 ]
    //     [ 3 3 2 ]
    //     [ 0 0 0 ]
    blob_bottom_label_->mutable_cpu_data()[0] = 1;
    blob_bottom_label_->mutable_cpu_data()[1] = 1;
    blob_bottom_label_->mutable_cpu_data()[2] = 0;
    blob_bottom_label_->mutable_cpu_data()[3] = 1;
    blob_bottom_label_->mutable_cpu_data()[4] = 0;
    blob_bottom_label_->mutable_cpu_data()[5] = 2;
    blob_bottom_label_->mutable_cpu_data()[6] = 2;
    blob_bottom_label_->mutable_cpu_data()[7] = 2;
    blob_bottom_label_->mutable_cpu_data()[8] = 0;
    blob_bottom_label_->mutable_cpu_data()[9] = 1;
    blob_bottom_label_->mutable_cpu_data()[10] = 2;
    blob_bottom_label_->mutable_cpu_data()[11] = 2;
    blob_bottom_label_->mutable_cpu_data()[12] = 3;
    blob_bottom_label_->mutable_cpu_data()[13] = 3;
    blob_bottom_label_->mutable_cpu_data()[14] = 2;
    blob_bottom_label_->mutable_cpu_data()[15] = 0;
    blob_bottom_label_->mutable_cpu_data()[16] = 0;
    blob_bottom_label_->mutable_cpu_data()[17] = 0;
  }

  Blob<Dtype>* const blob_bottom_prediction_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int num_labels_;
};

TYPED_TEST_CASE(ParseEvaluateLayerTest, TestDtypes);

TYPED_TEST(ParseEvaluateLayerTest, TestSetup) {
  LayerParameter layer_param;
  ParseEvaluateParameter* parse_evaluate_param = layer_param.mutable_parse_evaluate_param();
  parse_evaluate_param->set_num_labels(10);
  ParseEvaluateLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 10);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(ParseEvaluateLayerTest, TestForwardCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  ParseEvaluateParameter* parse_evaluate_param = layer_param.mutable_parse_evaluate_param();
  parse_evaluate_param->set_num_labels(5);
  ParseEvaluateLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Output:
  //     [ 2 6 3 ]
  //     [ 3 4 5 ]
  //     [ 3 6 5 ]
  //     [ 2 2 4 ]
  //     [ 0 0 1 ]
  EXPECT_EQ(this->blob_top_->cpu_data()[0], 2);
  EXPECT_EQ(this->blob_top_->cpu_data()[1], 6);
  EXPECT_EQ(this->blob_top_->cpu_data()[2], 3);
  EXPECT_EQ(this->blob_top_->cpu_data()[3], 3);
  EXPECT_EQ(this->blob_top_->cpu_data()[4], 4);
  EXPECT_EQ(this->blob_top_->cpu_data()[5], 5);
  EXPECT_EQ(this->blob_top_->cpu_data()[6], 3);
  EXPECT_EQ(this->blob_top_->cpu_data()[7], 6);
  EXPECT_EQ(this->blob_top_->cpu_data()[8], 5);
  EXPECT_EQ(this->blob_top_->cpu_data()[9], 2);
  EXPECT_EQ(this->blob_top_->cpu_data()[10], 2);
  EXPECT_EQ(this->blob_top_->cpu_data()[11], 4);
  EXPECT_EQ(this->blob_top_->cpu_data()[12], 0);
  EXPECT_EQ(this->blob_top_->cpu_data()[13], 0);
  EXPECT_EQ(this->blob_top_->cpu_data()[14], 1);
}

TYPED_TEST(ParseEvaluateLayerTest, TestForwardIgnoreCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  ParseEvaluateParameter* parse_evaluate_param = layer_param.mutable_parse_evaluate_param();
  parse_evaluate_param->set_num_labels(5);
  parse_evaluate_param->add_ignore_label(0);
  ParseEvaluateLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Output:
  //     [ 0 0 1 ]
  //     [ 3 4 4 ]
  //     [ 3 6 3 ]
  //     [ 2 2 4 ]
  //     [ 0 0 0 ]
  EXPECT_EQ(this->blob_top_->cpu_data()[0], 0);
  EXPECT_EQ(this->blob_top_->cpu_data()[1], 0);
  EXPECT_EQ(this->blob_top_->cpu_data()[2], 1);
  EXPECT_EQ(this->blob_top_->cpu_data()[3], 3);
  EXPECT_EQ(this->blob_top_->cpu_data()[4], 4);
  EXPECT_EQ(this->blob_top_->cpu_data()[5], 4);
  EXPECT_EQ(this->blob_top_->cpu_data()[6], 3);
  EXPECT_EQ(this->blob_top_->cpu_data()[7], 6);
  EXPECT_EQ(this->blob_top_->cpu_data()[8], 3);
  EXPECT_EQ(this->blob_top_->cpu_data()[9], 2);
  EXPECT_EQ(this->blob_top_->cpu_data()[10], 2);
  EXPECT_EQ(this->blob_top_->cpu_data()[11], 4);
  EXPECT_EQ(this->blob_top_->cpu_data()[12], 0);
  EXPECT_EQ(this->blob_top_->cpu_data()[13], 0);
  EXPECT_EQ(this->blob_top_->cpu_data()[14], 0);
}

TYPED_TEST(ParseEvaluateLayerTest, TestForwardIgnoreTwoCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  ParseEvaluateParameter* parse_evaluate_param = layer_param.mutable_parse_evaluate_param();
  parse_evaluate_param->set_num_labels(5);
  parse_evaluate_param->add_ignore_label(0);
  parse_evaluate_param->add_ignore_label(4);
  ParseEvaluateLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Output:
  //     [ 0 0 1 ]
  //     [ 3 4 4 ]
  //     [ 3 6 3 ]
  //     [ 2 2 4 ]
  //     [ 0 0 0 ]
  EXPECT_EQ(this->blob_top_->cpu_data()[0], 0);
  EXPECT_EQ(this->blob_top_->cpu_data()[1], 0);
  EXPECT_EQ(this->blob_top_->cpu_data()[2], 1);
  EXPECT_EQ(this->blob_top_->cpu_data()[3], 3);
  EXPECT_EQ(this->blob_top_->cpu_data()[4], 4);
  EXPECT_EQ(this->blob_top_->cpu_data()[5], 4);
  EXPECT_EQ(this->blob_top_->cpu_data()[6], 3);
  EXPECT_EQ(this->blob_top_->cpu_data()[7], 6);
  EXPECT_EQ(this->blob_top_->cpu_data()[8], 3);
  EXPECT_EQ(this->blob_top_->cpu_data()[9], 2);
  EXPECT_EQ(this->blob_top_->cpu_data()[10], 2);
  EXPECT_EQ(this->blob_top_->cpu_data()[11], 4);
  EXPECT_EQ(this->blob_top_->cpu_data()[12], 0);
  EXPECT_EQ(this->blob_top_->cpu_data()[13], 0);
  EXPECT_EQ(this->blob_top_->cpu_data()[14], 0);
}

}  // namespace caffe
