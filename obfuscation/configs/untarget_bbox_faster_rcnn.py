_base_ = [
    "./models/faster_rcnn.py",
    "./dataset/coco.py",
    "./runtime/untarget_bbox.py",
]

backward_loss = "get_faster_rcnn_untarget_loss"
