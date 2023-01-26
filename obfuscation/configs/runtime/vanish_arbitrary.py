_base_ = ["./vanish_bbox.py"]

bbox_sampler = "arbitrary"

# boundary_distance int: distance between arbitrary bbox and target bbox in original image pixel units
# arbitrary_bbox_length int: arbitrary bbox width and height in original image pixel units
perturb_kwargs = dict(arbitrary_bbox_length=96, boundary_distance=32)
