import numpy as np
from rangenet.tasks.semantic.modules.ioueval import iouEval

def compute_seg_accuracy(seg_model, synth_data, gt_labels, ignore=[]):
    pred, _ = seg_model(synth_data)
    pred = pred.argmax(dim=1)
    gt_labels = gt_labels.long()
    eval = iouEval(seg_model.nclasses, pred.get_device(), ignore=ignore)
    eval.addBatch(pred, gt_labels)
    _, iou = eval.getIoU()
    m_acc = eval.getacc()
    return iou, m_acc
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
