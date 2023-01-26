_base_ = [
    "./models/faster_rcnn.py",
    "./dataset/coco.py",
    "./runtime/mislabel_arbitrary.py",
]

backward_loss = "get_faster_rcnn_mislabel_loss"
