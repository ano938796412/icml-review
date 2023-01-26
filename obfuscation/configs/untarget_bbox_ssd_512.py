_base_ = [
    "./models/ssd_512.py",
    "./dataset/coco.py",
    "./runtime/untarget_bbox.py",
]

backward_loss = "get_ssd_512_untarget_loss"
