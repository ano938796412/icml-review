_base_ = ["./models/cascade_rcnn.py", "./dataset/coco.py", "./runtime/vanish_test.py"]

backward_loss = "get_cascade_rcnn_vanish_loss"
