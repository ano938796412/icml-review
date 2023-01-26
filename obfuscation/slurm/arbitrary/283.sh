python ./src/run.py --config configs/untarget_arbitrary_retinanet.py --log bbox_50_dist_50_repeat_1 --cfg-options perturb_kwargs.arbitrary_bbox_length=50 perturb_kwargs.boundary_distance=50 result_dir=./data/arbitrary/results dataset_dir=./data/arbitrary/datasets log_dir=./data/arbitrary/logs img_dir=./data/arbitrary/images attack_samples=100 itrs=200 seed=19116 shuffle=True
