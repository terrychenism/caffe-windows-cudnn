# You should change $CAFFE_ROOT to the caffe root directory (e.g. ~/project/caffe) or make it as environment variable (e.g. put it in ~/.bashrc)

# if [ ! -d jobs/VGGNet/VOC2012ext ]
# then
#   mkdir -p jobs/VGGNet/VOC2012ext
# fi

mkdir snapshots

../../build/tools/caffe train \
--solver="solver.prototxt" \
--weights="fcn-8s-pascalcontext.caffemodel" \
--gpu 3 2>&1 | tee crfasrnn_pascal_context.log


#--weights="../../models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
#--weights="../../models/VGGNet/VGG_ILSVRC_16_layers_conv.caffemodel" \
#--snapshot="snapshots/VGG_context2_iter_16000.solverstate" \

#--weights="fcn-8s-pascalcontext.caffemodel" \