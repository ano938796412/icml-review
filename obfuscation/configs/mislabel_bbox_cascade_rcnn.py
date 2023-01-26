_base_ = ["./models/cascade_rcnn.py", "./dataset/coco.py", "./runtime/mislabel_bbox.py"]

backward_loss = "get_cascade_rcnn_mislabel_loss"
