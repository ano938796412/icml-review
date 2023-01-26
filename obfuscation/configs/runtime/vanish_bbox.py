# dataset
gt_samples = 5000

attack_samples = None
shuffle = False

seed = None
replace_dataset = True

compute_map = False

# device
cuda = 0

# app
launch_app = False

# attack
itrs = (10, 50, 100, 200)
min_iou = 0.3
min_score = 0.3

adversarial_target = "vanish"
bbox_sampler = "non_overlapping"
attack_bbox = "ground_truth"
perturb_kwargs = dict(perturb_fun="perturb_inside")

# data
result_dir = "./data/run/results"
dataset_dir = "./data/run/datasets"

log_dir = "./data/run/logs"

img_dir = "./data/run/images"
viz_pgd = False

# misc
log_level = "DEBUG"
