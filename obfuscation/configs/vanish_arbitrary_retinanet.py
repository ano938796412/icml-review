_base_ = [
    "./models/retinanet.py",
    "./dataset/coco.py",
    "./runtime/vanish_arbitrary.py",
]

backward_loss = "get_retinanet_vanish_loss"
