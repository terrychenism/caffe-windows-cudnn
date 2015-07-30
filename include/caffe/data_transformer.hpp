#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class DataTransformer {
 public:
  explicit DataTransformer(const TransformationParameter& param, Phase phase);
  virtual ~DataTransformer() {}

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();

  void FillInOffsets(int *w, int *h, int width, int height, int crop_size) {
	  FillInOffsets(w, h, width, height, crop_size, crop_size);
	  // w[0] = 0; h[0] = 0;
	  // w[1] = 0; h[1] = height - crop_size;
	  // w[2] = width - crop_size; h[2] = 0;
	  // w[3] = width - crop_size; h[3] = height - crop_size;
	  // w[4] = (width - crop_size) / 2; h[4] = (height - crop_size) / 2;
  }

  void FillInOffsets(int *w, int *h, int width, int height, int crop_w, int crop_h) {
	  if (crop_w < width * 2 / 3 && crop_h < height * 2 / 3) {
		  // we want to be conservative when the crop is small
		  w[0] = 0; h[0] = (height - crop_h) / 2;
		  w[1] = width - crop_w; h[1] = (height - crop_h) / 2;
		  w[2] = (width - crop_w) / 2; h[2] = 0;
		  w[3] = (width - crop_w) / 2; h[3] = height - crop_h;
		  w[4] = (width - crop_w) / 2; h[4] = (height - crop_h) / 2;
	  }
	  else {
		  w[0] = 0; h[0] = 0;
		  w[1] = 0; h[1] = height - crop_h;
		  w[2] = width - crop_w; h[2] = 0;
		  w[3] = width - crop_w; h[3] = height - crop_h;
		  w[4] = (width - crop_w) / 2; h[4] = (height - crop_h) / 2;
	  }
  }

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
  void Transform(const Datum& datum, Blob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Datum.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<Datum> & datum_vector,
                Blob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Mat.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<cv::Mat> & mat_vector,
                Blob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   */
  void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob);

  /**
   * @brief Applies the same transformation defined in the data layer's
   * transform_param block to all the num images in a input_blob.
   *
   * @param input_blob
   *    A Blob containing the data to be transformed. It applies the same
   *    transformation to all the num images in the blob.
   * @param transformed_blob
   *    This is destination blob, it will contain as many images as the
   *    input blob. It can be part of top blob's data.
   */
  void Transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob);

  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const Datum& datum);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const vector<Datum> & datum_vector);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   */
  vector<int> InferBlobShape(const vector<cv::Mat> & mat_vector);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   */
  vector<int> InferBlobShape(const cv::Mat& cv_img);


  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param batch_item_id
   *    Datum position within the batch. This is used to compute the
   *    writing position in the top blob's data
   * @param datum
   *    Datum containing the data to be transformed.
   * @param mean
   * @param transformed_data
   *    This is meant to be the top blob's data. The transformed data will be
   *    written at the appropriate place within the blob's data.
   */
  void Transform(const int batch_item_id, const Datum& datum,
                 const Dtype* mean, Dtype* transformed_data);
  void Transform(const int batch_item_id, IplImage *img,
                 const Dtype* mean, Dtype* transformed_data);


 protected:
   /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  virtual int Rand(int n);
 /* virtual int Rand();*/

  virtual float Uniform(const float min, const float max);

  void Transform(const Datum& datum, Dtype* transformed_data);

  void TransformSingle(const int batch_item_id, IplImage *img,
	  const Dtype* mean, Dtype* transformed_data);
  void TransformMultiple(const int batch_item_id, IplImage *img,
	  const Dtype* mean, Dtype* transformed_data);
  
  // Tranformation parameters
  TransformationParameter param_;


  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_

