_base_ = ["./models/ssd_512.py", "./dataset/coco.py", "./runtime/vanish_bbox.py"]

backward_loss = "get_ssd_512_vanish_loss"
