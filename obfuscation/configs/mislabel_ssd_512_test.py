_base_ = ["./models/ssd_512.py", "./dataset/coco.py", "./runtime/mislabel_test.py"]

backward_loss = "get_ssd_512_mislabel_loss"
