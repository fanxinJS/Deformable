./tools/train_net.py --gpu 0 \
  --solver models/pascal_voc/ResNet-50/rfcn_end2end/solver_ohem.prototxt \
  --snapshot data/resnet50_rfcn_ohem_iter_60000.solverstate \
  --iters 120000 \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/rfcn_end2end_ohem.yml \



