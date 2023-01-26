_base_ = ["./vanish_bbox.py"]

# _base_ = ["./vanish_arbitrary.py"]
# perturb_kwargs = dict(arbitrary_bbox_length=100, boundary_distance=10)

gt_samples = 250
attack_samples = 50

shuffle = True
seed = 5151

launch_app = True

itrs = (10, 1)
attack_bbox = "predictions" # "ground_truth"

result_dir = "./data/run_test/results"
dataset_dir = "./data/run_test/datasets"

log_dir = "./data/run_test/logs"

img_dir = "./data/run_test/images"
viz_pgd = False
