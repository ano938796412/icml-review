_base_ = [
    "./models/cascade_rcnn.py",
    "./dataset/coco.py",
    "./runtime/untarget_arbitrary.py",
]

backward_loss = "get_cascade_rcnn_untarget_loss"
