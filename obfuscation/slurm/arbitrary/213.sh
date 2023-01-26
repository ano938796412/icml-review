python ./src/run.py --config configs/vanish_arbitrary_retinanet.py --log bbox_100_dist_200_repeat_1 --cfg-options perturb_kwargs.arbitrary_bbox_length=100 perturb_kwargs.boundary_distance=200 result_dir=./data/arbitrary/results dataset_dir=./data/arbitrary/datasets log_dir=./data/arbitrary/logs img_dir=./data/arbitrary/images attack_samples=100 itrs=200 seed=13422 shuffle=True
