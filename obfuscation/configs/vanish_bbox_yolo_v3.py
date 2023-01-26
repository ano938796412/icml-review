_base_ = ["./models/yolo_v3.py", "./dataset/coco.py", "./runtime/vanish_bbox.py"]

backward_loss = "get_yolo_v3_vanish_loss"
