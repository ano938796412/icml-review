_base_ = ["./models/retinanet.py", "./dataset/coco.py", "./runtime/mislabel_test.py"]

backward_loss = "get_retinanet_mislabel_loss"
