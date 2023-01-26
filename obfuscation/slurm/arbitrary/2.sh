python ./src/run.py --config configs/vanish_arbitrary_yolo_v3.py --log bbox_200_dist_10_repeat_1 --cfg-options perturb_kwargs.arbitrary_bbox_length=200 perturb_kwargs.boundary_distance=10 result_dir=./data/arbitrary/results dataset_dir=./data/arbitrary/datasets log_dir=./data/arbitrary/logs img_dir=./data/arbitrary/images attack_samples=100 itrs=200 seed=11811 shuffle=True
