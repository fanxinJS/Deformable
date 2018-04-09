python tools/test_net.py --gpu 0 \
  --def models/pascal_voc/ResNet-50/rfcn_end2end/test_agnostic.prototxt \
  --net output/rfcn_end2end_ohem/voc_2007_trainval/resnet50_rfcn_ohem_iter_100000.caffemodel \
  --cfg experiments/cfgs/rfcn_end2end_ohem.yml \
  --imdb voc_2007_test \
##--set TEST.SOFT_NMS 1 \
  ${EXTRA_ARGS}
