_base_ = ["./models/retinanet.py", "./dataset/coco.py", "./runtime/vanish_test.py"]

backward_loss = "get_retinanet_vanish_loss"
