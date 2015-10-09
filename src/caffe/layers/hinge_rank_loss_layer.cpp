/*!
 *  \brief     The Caffe layer that implements the hinge rank loss described in the paper:
 *  \version   1.0
 *  \date      2015
 *  \details   If you use this code, please consider citing our paper:
 *             
 */

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
	void HingeRankLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		LossLayer<Dtype>::LayerSetUp(bottom, top);
		//const int class_num = 2;
		//const int label_dim = 300;
		//word2vec.resize(class_num); 
		//vector<vector<double>> labels = { { -0.028949, -0.036674, -0.015775, 0.437298, -0.088639, -0.042918, 0.015891, 0.127036, -0.040084, 0.295416, -0.148570, 0.342298, -0.089213, -0.191414, 0.032168, 0.229498, 0.333617, -0.266235, 0.024776, -0.201635, -0.021318, -0.133844, 0.230262, -0.106119, 0.057988, -0.240130, 0.046122, -0.158201, -0.240305, -0.042415 },
		//{ -0.225061, -0.198552, 0.097641, 0.305513, -0.219246, -0.027500, 0.046315, 0.192291, -0.169128, 0.317446, -0.213247, 0.333882, 0.057664, -0.068329, -0.043419, 0.108704, 0.088895, -0.168517, 0.157355, -0.143428, -0.078658, -0.184941, 0.078470, -0.014285, 0.120626, -0.335926, 0.210400, -0.214106, -0.101988, -0.234073 } };

		//string filename = "par.w2v";
		string filename = this->layer_param_.hinge_rank_loss_param().word2vec();
		LOG(INFO) << "Read label from " << filename;
		std::fstream labelFile(filename.c_str(), std::ios_base::in);

		//int class_num = 0;
		//int label_dim = 0;
		labelFile >> label_dim;
		labelFile >> class_num;
		LOG(INFO) << "(class_num, label_dim, margin) = (" << class_num << " " << label_dim << " " << this->layer_param_.hinge_rank_loss_param().margin() << ")";

		word2vec.resize(class_num);

		Dtype label;
		for (int i = 0; i < class_num; i++){
			word2vec[i] = new Dtype[label_dim];
			for (int j = 0; j < label_dim; j++){
				labelFile >> label;
				word2vec[i][j] = label;
				//std::cout << label << " ";
				//word2vec[i][j] = static_cast<Dtype>(labels[i][j]);
				//LOG(INFO) << word2vec[i][j];
			}
			//std::cout << std::endl;
		}
		labelFile.close();

		int num = bottom[0]->num();
		marginWithLoss_.resize(num);

	}

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

		//const int class_num = 2;
		//const int label_dim = 300;
		CHECK_EQ(label_dim, dim);

		CHECK_EQ(word2vec.size(), class_num);

		Dtype *query_similar = new Dtype[num];
		vector<Dtype*> query_dissimilar(num);
		Dtype* loss = top[0]->mutable_cpu_data();
		loss[0] = 0;

		for (int i = 0; i < num; ++i){

			marginWithLoss_[i].resize(word2vec.size());
			query_dissimilar[i] = new Dtype[class_num];
			Dtype *similar_label = word2vec[label[i]];
			query_similar[i] = caffe_cpu_dot(dim, &bottom_data[i*dim], similar_label /*&similar_label[i*dim]*/);
			for (int j = 0; j < word2vec.size(); j++){
				if (j != label[i]){
					Dtype *dissimilar_label = word2vec[j];
					query_dissimilar[i][j] = caffe_cpu_dot(dim, &bottom_data[i*dim], dissimilar_label/*&dissimilar_label[i*dim]*/);


					//avg_loss += (query_similar[i] - query_dissimilar[i][j]);

					// if(std::isnan(avg_loss)) 
					//  {
					//  	std:: cout << "nan" << std::endl;
					//  	for(int k = 0; k<dim; ++k)
					//  	{
					//  		if(std::isnan(bottom_data[i*dim+k])) 
					//  		    std:: cout << "bottom_data[" << k <<"] is nan" << std::endl;  
					//  	}
					//  	if(std::isnan(query_similar[i])) 
					//  	   std:: cout << "query_similar[i] is nan" << std::endl;
					//  	if(std::isnan(query_dissimilar[i][j])) 
					//  		std:: cout << "query_dissimilar[i][j] is nan" << std::endl;
					//  }
					marginWithLoss_[i][j] = margin - query_similar[i] + query_dissimilar[i][j]; // C 10, dim (marginWithLoss_) = N *( C-1 )
				}
				//else
				//marginWithLoss_[i][j] = 0;

				loss[0] += std::max(Dtype(0), marginWithLoss_[i][j]);
			}
			delete[] query_dissimilar[i];
		}

		loss[0] /= num;
		delete[] query_similar;
		query_dissimilar.clear();


		// ==================== hinge loss ===================== 
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
		loss[0] = caffe_cpu_asum(count, bottom_diff) / num; */
	}

	template <typename Dtype>
	void HingeRankLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


		if (propagate_down[0]) {
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			const Dtype* label = bottom[1]->cpu_data();
			int num = bottom[0]->num();
			int count = bottom[0]->count();
			int dim = count / num;

			for (int i = 0; i < num; ++i) {
				Dtype *similar_label = word2vec[static_cast<int>(label[i])];
				for (int j = 0; j < marginWithLoss_[i].size(); j++){
					Dtype *dissimilar_label = word2vec[j];
					if (marginWithLoss_[i][j] > 0){
						for (int idx = 0; idx < dim; idx++)
							bottom_diff[i * dim + idx] += -similar_label[idx] + dissimilar_label[idx];
					}
				}
			}

			const Dtype loss_weight = top[0]->cpu_diff()[0];
			// caffe_cpu_sign(count, bottom_diff, bottom_diff);
			caffe_scal(count, loss_weight / num, bottom_diff);

		}

		// ==================== hinge loss ======================
		/*if (propagate_down[1]) {
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
		}*/
	}

	INSTANTIATE_CLASS(HingeRankLossLayer);
	REGISTER_LAYER_CLASS(HingeRankLoss);

}  // namespace caffe
