python ./src/run.py --config configs/mislabel_arbitrary_retinanet.py --log bbox_100_dist_50_repeat_2 --cfg-options perturb_kwargs.arbitrary_bbox_length=100 perturb_kwargs.boundary_distance=50 result_dir=./data/arbitrary/results dataset_dir=./data/arbitrary/datasets log_dir=./data/arbitrary/logs img_dir=./data/arbitrary/images attack_samples=100 itrs=200 seed=31218 shuffle=True
