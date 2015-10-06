#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void HingeRankLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  Dtype margin = this->layer_param_.hinge_rank_loss_param().margin();

  const int class_num = 10;
  const int label_dim = 30;
  CHECK_EQ(label_dim, dim);
  vector<Dtype*> word2vec;

  Dtype *query_similar = new Dtype[num];
  vector<Dtype*> query_dissimilar(num);
  Dtype* loss = top[0]->mutable_cpu_data();

  for (int i = 0; i < num; ++i){
	query_dissimilar[i] = new Dtype[label_dim-1];
    Dtype *similar_label = word2vec[label[i]];
	query_similar[i] = caffe_cpu_dot(dim, &bottom_data[i*dim], &similar_label[i*dim]);
	for (int j = 0; j != label[i] && j < word2vec.size(); j++){
		Dtype *dissimilar_label = word2vec[j];
		query_dissimilar[i][j] = caffe_cpu_dot(dim, &bottom_data[i*dim], &dissimilar_label[i*dim]);

		marginWithLoss_[i][j] = margin - query_similar[i] + query_dissimilar[i][j]; // C 10, dim (marginWithLoss_) = N *( C-1 )
		loss[0] += std::max(Dtype(0), marginWithLoss_[i][j]);
	}
	delete[] query_dissimilar[i];
  }



  loss[0] /= num;
  delete[] query_similar;
  query_dissimilar.clear();
  
  /*caffe_copy(count, bottom_data, bottom_diff);
  for (int i = 0; i < num; ++i) {
    bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
  }
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      bottom_diff[i * dim + j] = std::max(
        Dtype(0), 1 + bottom_diff[i * dim + j]);
    }
  }
  Dtype* loss = top[0]->mutable_cpu_data();
  loss[0] = caffe_cpu_asum(count, bottom_diff) / num;*/
}

template <typename Dtype>
void HingeRankLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    for (int i = 0; i < num; ++i) {
      bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
    }

    const Dtype loss_weight = top[0]->cpu_diff()[0];
	caffe_cpu_sign(count, bottom_diff, bottom_diff);
	caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_CLASS(HingeRankLossLayer);
REGISTER_LAYER_CLASS(HingeRankLoss);

}  // namespace caffe
