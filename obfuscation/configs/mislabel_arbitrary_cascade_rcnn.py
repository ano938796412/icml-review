_base_ = [
    "./models/cascade_rcnn.py",
    "./dataset/coco.py",
    "./runtime/mislabel_arbitrary.py",
]

backward_loss = "get_cascade_rcnn_mislabel_loss"
