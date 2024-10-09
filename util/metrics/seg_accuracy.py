import numpy as np
from rangenet.tasks.semantic.modules.ioueval import iouEval
import torchvision.transforms.functional as TF
from util import _map

def compute_seg_accuracy(seg_model, synth_data, gt_labels, ignore=[], label_map=None, fetched_mask=None):
    # H , W = 64, 2048
    # synth_data = TF.resize(synth_data, (H, W), TF.InterpolationMode.NEAREST)
    # gt_labels = TF.resize(gt_labels, (H, W), TF.InterpolationMode.NEAREST)
    pred, _ = seg_model(synth_data)
    pred = pred.argmax(dim=1)
    if fetched_mask is not None:
        pred = pred * fetched_mask.squeeze()
    if label_map is not None:
        pred = _map(pred, label_map)
    gt_labels = gt_labels.long()
    eval = iouEval(seg_model.nclasses, pred.get_device(), ignore=ignore)
    eval.addBatch(pred, gt_labels)
    _, iou = eval.getIoU()
    prec, rec = eval.getPreRec()
    m_acc = eval.getacc()
    return iou, m_acc, prec, rec
    # pred_np = pred.cpu().numpy()
    # pred_np = pred_np.reshape((-1)).astype(np.int32)

    # if seg_model.post:
    #     # knn postproc
    #     unproj_argmax = seg_model.post(proj_range,
    #                             unproj_range,
    #                             proj_argmax,
    #                             p_x,
    #                             p_y)
    # else:
    #     # put in original pointcloud using indexes
