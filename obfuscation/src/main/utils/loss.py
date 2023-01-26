"""pgd.py does gradient descent on the image to minimize the loss.
To increase untarget loss, we have to return the negative loss"""


# faster_rcnn losses dict_keys(['loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox'])
# where only rpn losses is List[Tensor] given at every FPN scale
# (default is 5 in mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py:18)
def get_faster_rcnn_vanish_loss(losses):
    return (
        sum(losses["loss_rpn_cls"]) + losses["loss_cls"]
    )  # obj + cls (include background)


def get_faster_rcnn_mislabel_loss(losses):
    return losses["loss_cls"]  # cls only


def get_faster_rcnn_untarget_loss(losses):
    l = (
        sum(losses["loss_rpn_cls"])
        + sum(losses["loss_rpn_bbox"])
        + losses["loss_cls"]
        + losses["loss_bbox"]
    )
    return -l


# yolo_v3 losses dict_keys(['loss_cls', 'loss_conf', 'loss_xy', 'loss_wh'])
# where every loss is List[Tensor] given at every scale
# (default is 3 in mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py:12)
def get_yolo_v3_vanish_loss(losses):
    return sum(losses["loss_conf"])  # obj prob only


def get_yolo_v3_mislabel_loss(losses):
    return sum(losses["loss_cls"])  # cls prob only


def get_yolo_v3_untarget_loss(losses):
    l = (
        sum(losses["loss_cls"])
        + sum(losses["loss_conf"])
        + sum(losses["loss_xy"])
        + sum(losses["loss_wh"])
    )

    return -l


# retinanet losses dict_keys(['loss_cls', 'loss_bbox'])
# where every loss is List[Tensor] given at every FPN scale
# (default is 5 mmdetection/configs/_base_/models/retinanet_r50_fpn.py:20)
def get_retinanet_vanish_loss(losses):
    return sum(losses["loss_cls"])  # cls (include background)


def get_retinanet_mislabel_loss(losses):
    return get_retinanet_vanish_loss(losses)  # cls (include background)


def get_retinanet_untarget_loss(losses):
    l = sum(losses["loss_cls"]) + sum(losses["loss_bbox"])
    return -l


# ssd_512 losses dict_keys(['loss_cls', 'loss_bbox'])
# where every loss is List[Tensor] with len 1
def get_ssd_512_vanish_loss(losses):
    return sum(losses["loss_cls"])  # cls (include background)


def get_ssd_512_mislabel_loss(losses):
    return get_ssd_512_vanish_loss(losses)  # cls (include background)


def get_ssd_512_untarget_loss(losses):
    l = sum(losses["loss_cls"]) + sum(losses["loss_bbox"])
    return -l


# cascade_rcnn losses dict_keys(['loss_rpn_cls', 'loss_rpn_bbox',
# 's0.loss_cls', 's0.loss_bbox', 's1.loss_cls', 's1.loss_bbox', 's2.loss_cls', 's2.loss_bbox'])
# where only rpn losses is List[Tensor] given at every FPN scale
# (default is 5 in mmdetection/configs/_base_/models/cascade_rcnn_r50_fpn.py:18)
# and roi losses are Tensor at 3 cascade scales
def get_cascade_rcnn_vanish_loss(losses):
    return sum(losses["loss_rpn_cls"]) + sum(
        [losses[k] for k in losses.keys() if ".loss_cls" in k]
    )  # obj + cls (include background)


def get_cascade_rcnn_mislabel_loss(losses):
    return sum([losses[k] for k in losses.keys() if ".loss_cls" in k])  # cls only


def get_cascade_rcnn_untarget_loss(losses):
    l = (
        sum(losses["loss_rpn_cls"])
        + sum(losses["loss_rpn_bbox"])
        + sum([losses[k] for k in losses.keys() if ".loss_cls" in k])
        + sum([losses[k] for k in losses.keys() if ".loss_bbox" in k])
    )
    return -l
