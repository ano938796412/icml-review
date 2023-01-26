python ./src/run.py --config configs/mislabel_arbitrary_cascade_rcnn.py --log bbox_50_dist_200_repeat_2 --cfg-options perturb_kwargs.arbitrary_bbox_length=50 perturb_kwargs.boundary_distance=200 result_dir=./data/arbitrary/results dataset_dir=./data/arbitrary/datasets log_dir=./data/arbitrary/logs img_dir=./data/arbitrary/images attack_samples=100 itrs=200 seed=17688 shuffle=True