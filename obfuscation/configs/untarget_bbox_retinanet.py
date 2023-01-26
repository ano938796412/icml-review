_base_ = [
    "./models/retinanet.py",
    "./dataset/coco.py",
    "./runtime/untarget_bbox.py",
]

backward_loss = "get_retinanet_untarget_loss"
