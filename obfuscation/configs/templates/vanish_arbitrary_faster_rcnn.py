_base_ = [
    "./models/faster_rcnn.py",
    "./dataset/coco.py",
    "./runtime/vanish_arbitrary.py",
]

backward_loss = "get_faster_rcnn_vanish_loss"
