#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EvaluateLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	string filename = this->layer_param_.evaluate_param().word2vec();
	LOG(INFO) << "Read label from " << filename;
	std::fstream labelFile(filename.c_str(), std::ios_base::in);

	labelFile >> label_dim;
	labelFile >> class_num;
	LOG(INFO) << "(class_num, label_dim) = (" << class_num << " " << label_dim << ")";

	word2vec.resize(class_num);

	Dtype label;
	for (int i = 0; i < class_num; i++){
		word2vec[i] = new Dtype[label_dim];
		for (int j = 0; j < label_dim; j++){
			labelFile >> label;
			word2vec[i][j] = label;
		}
	}
	labelFile.close();

	const EvaluateParameter& parse_evaluate_param = this->layer_param_.evaluate_param();
	CHECK(parse_evaluate_param.has_num_labels()) << "Must have num_labels!!";
	num_labels_ = parse_evaluate_param.num_labels();
	CHECK_EQ(num_labels_, class_num);

	ignore_labels_.clear();
	int num_ignore_label = parse_evaluate_param.ignore_label().size();
	for (int i = 0; i < num_ignore_label; ++i) {
		ignore_labels_.insert(parse_evaluate_param.ignore_label(i));
	}

}

template <typename Dtype>
void EvaluateLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void EvaluateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

	Dtype correct = 0;
	int num = bottom[0]->num(); // batch size
	int count = bottom[0]->count();
	int dim = count / num;
	CHECK_EQ(label_dim, dim);
	CHECK_EQ(word2vec.size(), class_num);
	const Dtype* bottom_pred = bottom[0]->cpu_data(); // predict
	const Dtype* bottom_gt = bottom[1]->cpu_data();   // label num*1
	//Dtype* top_data = top[0]->mutable_cpu_data();     // output accuracy
	//caffe_set(top[0]->count(), Dtype(0), top_data);
	
	for (int i = 0; i < num; ++i) {

		int gt_label = bottom_gt[i];
		Dtype max_dot = INT_MIN;
		int max_label;
		for (int j = 0; j < word2vec.size(); j++){
			Dtype *word2vec_label = word2vec[j];
			Dtype pred_dot = caffe_cpu_dot(dim, &bottom_pred[i*dim], word2vec_label);
			if (pred_dot > max_dot){
				max_dot = pred_dot;
				max_label = j;
			}
		}
		//LOG(INFO) << "predict: " << max_label << "  VS  GT: " << gt_label;
		if (gt_label == max_label)
			correct++;
		
	}
	//LOG(INFO) << "Accuracy: " << (Dtype)correct/num; 
	top[0]->mutable_cpu_data()[0] = correct / num;
}

INSTANTIATE_CLASS(EvaluateLayer);
REGISTER_LAYER_CLASS(Evaluate);

}  // namespace caffe
