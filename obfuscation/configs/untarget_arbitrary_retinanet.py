_base_ = [
    "./models/retinanet.py",
    "./dataset/coco.py",
    "./runtime/untarget_arbitrary.py",
]

backward_loss = "get_retinanet_untarget_loss"
