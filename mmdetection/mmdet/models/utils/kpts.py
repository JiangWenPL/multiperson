import torch
from torch import nn
import torch.functional as F
import numpy as np
import mmcv
import cv2


def kpts_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_kpts_list,
                cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(kpts_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_kpts_list, cfg_list)
    mask_targets = torch.cat(list(mask_targets))
    return mask_targets


@torch.no_grad()
def kpts_target_single(pos_proposals, pos_assigned_gt_inds, gt_kpts, cfg):
    heatmap_size = cfg.heatmap_size
    num_pos = pos_proposals.size(0)
    mask_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            cur_gt_kpts = gt_kpts[pos_assigned_gt_inds[i]]
            bbox = proposals_np[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            target = np.zeros((cur_gt_kpts.shape[0], heatmap_size, heatmap_size))
            for k_id, kp in enumerate(cur_gt_kpts.cpu().numpy()):
                y_roi, x_roi = int(np.round(heatmap_size * (kp[1] - y1) / h)), int(
                    np.round(heatmap_size * (kp[0] - x1) / w))
                if kp[-1] != 0 and y_roi > 0 and x_roi > 0 and y_roi < heatmap_size and x_roi < heatmap_size:
                    target[k_id, y_roi, x_roi] = 1
                    target[k_id] = cv2.GaussianBlur(target[k_id], cfg.gamma, 0)
                    am = np.amax(target[k_id])
                    target[k_id] /= am
            mask_targets.append(target)
        mask_targets = torch.from_numpy(np.stack(mask_targets)).float().to(
            pos_proposals.device)
    else:
        mask_targets = pos_proposals.new_zeros((0, heatmap_size, heatmap_size))
    return mask_targets
