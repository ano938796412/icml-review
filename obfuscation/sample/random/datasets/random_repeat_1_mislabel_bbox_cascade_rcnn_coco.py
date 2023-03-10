model_config = '../mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
model_checkpoint = '../mmdetection/checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
dataset_name = 'random_repeat_1_mislabel_bbox_cascade_rcnn_coco'
images_path = '../coco/val2017'
labels_path = '../coco/annotations/instances_val2017.json'
gt_samples = 1000
attack_samples = 100
shuffle = True
seed = 8617
replace_dataset = True
compute_map = False
cuda = 0
launch_app = False
itrs = (10, 50, 100, 200)
min_iou = 0.3
min_score = 0.3
adversarial_target = 'mislabel'
bbox_sampler = 'non_overlapping'
attack_bbox = 'ground_truth'
perturb_kwargs = dict(perturb_fun='perturb_inside')
result_dir = './data/random/results'
dataset_dir = './data/random/datasets'
log_dir = './data/random/logs'
img_dir = './data/random/images'
viz_pgd = False
log_level = 'DEBUG'
backward_loss = 'get_cascade_rcnn_mislabel_loss'
log_name = 'random_repeat_1_mislabel_bbox_cascade_rcnn'
