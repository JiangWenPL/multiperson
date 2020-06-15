import mmcv.runner.hooks.logger as mmcv_logger
from mmcv.runner.utils import master_only
import os.path as osp
from mmdetection.mmdet.models.utils.smpl.renderer import Renderer
from mmdetection.mmdet.models.utils.smpl.viz import draw_skeleton, J24_TO_J14
import torchvision
import torch
import numpy as np
import cv2
import traceback
import math
import matplotlib.pyplot as plt
from itertools import zip_longest

to_np = lambda x: x[0].transpose([1, 2, 0])


class SMPLBoard(mmcv_logger.LoggerHook):
    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 detail_mode=False,
                 bboxes_only=False,
                 K_SMALLEST=10000,
                 draw_skeleton=False,
                 bird_view=True,
                 draw_rpn_bbox=False,
                 draw_det_bbox=True,
                 draw_gt_bbox=False,
                 aten_threshold=0.5,
                 FOCAL_LENGTH=1000,
                 ):
        super(SMPLBoard, self).__init__(interval, ignore_last,
                                        reset_flag)
        self.log_dir = log_dir
        self.focal_length = FOCAL_LENGTH
        if detail_mode:
            self.BATCHES_ITER = 10  # Number of batches to visualize at every interval
            self.MAX_IMAGES = 10  # Number of images to visualize in each batch
            self.N_ROWS = 4
            self.MIN_WIDTH = 1024
        else:
            self.BATCHES_ITER = 1  # Number of batches to visualize at every interval
            self.MAX_IMAGES = 4  # Number of images to visualize in each batch
            self.N_ROWS = 2
            self.MIN_WIDTH = 512
        self.resizer = torchvision.transforms.Compose(
            [torchvision.transforms.ToPILImage(), torchvision.transforms.Resize(self.MIN_WIDTH),
             torchvision.transforms.ToTensor()])
        self.K_SMALLEST = K_SMALLEST  # Select k smallest error proposals in images to visualize
        self.bboxes_only = bboxes_only
        self.draw_skeleton = draw_skeleton
        self.bird_view = bird_view
        self.draw_det_bbox = draw_det_bbox
        self.draw_gt_bbox = draw_gt_bbox
        self.draw_rpn_bbox = draw_rpn_bbox
        self.aten_threshold = aten_threshold

    @master_only
    def before_run(self, runner):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError('Please install tensorflow and tensorboardX '
                              'to use TensorboardLoggerHook.')
        else:
            if self.log_dir is None:
                self.log_dir = osp.join(runner.work_dir, 'smpl_logs')
            self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        self.render = Renderer(focal_length=self.focal_length)
        # Log learning rate in training mode
        if runner.mode == 'train':
            self.writer.add_scalar('lr', runner.current_lr()[0], runner.iter)

        for var in runner.log_buffer.output:
            if var in ['time', 'data_time'] or not var.endswith(f'/{runner.mode}'):
                continue
            # tag = '{}/{}'.format(var, runner.mode)
            tag = var
            # To avoid wrong loss plot for validation set.
            pseudo_iter = runner.iter + runner.inner_iter if runner.mode == 'val' else runner.iter
            record = runner.log_buffer.output[var]
            if isinstance(record, str):
                self.writer.add_text(tag, record, pseudo_iter)
            elif var.startswith('img$'):
                pass
            else:
                self.writer.add_scalar(tag, runner.log_buffer.output[var],
                                       pseudo_iter)
        try:
            with torch.no_grad():
                rendered_imgs = list()
                bird_view_rendered = list()
                aten_mask_imgs = list()
                heatmap_imgs = list()
                R_bv = torch.zeros(3, 3)
                R_bv[0, 0] = R_bv[2, 1] = 1
                R_bv[1, 2] = -1
                for raw_images, idxs_in_batch, pred_vertices, translation, error_ranks, pred_bboxes, pred_poses2d, gt_poses2d, pose_idx, head_bboxes, gt_bboxes, aten_masks, heatmaps in zip_longest(
                        runner.log_buffer.output[f'img$raw_images/{runner.mode}'][-self.BATCHES_ITER:],
                        runner.log_buffer.output[f'img$idxs_in_batch/{runner.mode}'][-self.BATCHES_ITER:],
                        runner.log_buffer.output[f'img$pred_vertices/{runner.mode}'][-self.BATCHES_ITER:],
                        runner.log_buffer.output[f'img$translation/{runner.mode}'][-self.BATCHES_ITER:],
                        runner.log_buffer.output[f'img$error_rank/{runner.mode}'][-self.BATCHES_ITER:],
                        runner.log_buffer.output[f'img$pred_bboxes/{runner.mode}'][-self.BATCHES_ITER:],
                        runner.log_buffer.output[f'img$pred_keypoints_2d_smpl/{runner.mode}'][-self.BATCHES_ITER:],
                        runner.log_buffer.output[f'img$gt_keypoints_2d/{runner.mode}'][-self.BATCHES_ITER:],
                        runner.log_buffer.output[f'img$pose_idx/{runner.mode}'][-self.BATCHES_ITER:],
                        runner.log_buffer.output[f'img$head_bboxes/{runner.mode}'][-self.BATCHES_ITER:],
                        runner.log_buffer.output[f'img$gt_bboxes/{runner.mode}'][-self.BATCHES_ITER:],
                        runner.log_buffer.output.get(f'img$aten_masks/{runner.mode}', [])[-self.BATCHES_ITER:],
                        runner.log_buffer.output.get(f'img$heatmaps/{runner.mode}', [])[-self.BATCHES_ITER:],

                ):
                    N_ROWS = 2
                    if aten_masks:
                        N_ROWS += 1
                    if heatmaps is not None:
                        N_ROWS += 1
                    self.N_ROWS = N_ROWS

                    batch_idxs = list(set(idxs_in_batch.squeeze().int().tolist()))

                    # Truncate to visualize limted number of images per batch
                    batch_idxs = batch_idxs[:self.MAX_IMAGES]
                    raw_images = raw_images[:self.MAX_IMAGES]

                    pred_vertices_render = [list() for _ in batch_idxs]
                    translation_render = [list() for _ in batch_idxs]
                    bboxes_render = [list() for _ in batch_idxs]
                    head_bboxes_render = [list() for _ in batch_idxs]
                    gt_bboxes_render = [list() for _ in batch_idxs]
                    pred_poses2d_render = [list() for _ in batch_idxs]
                    gt_poses2d_render = [list() for _ in batch_idxs]
                    aten_masks_render = [list() for _ in batch_idxs]
                    heatmaps_render = [list() for _ in batch_idxs]
                    idxs_np = idxs_in_batch.squeeze().int().cpu().numpy()
                    for bid in batch_idxs:
                        rel_idx_batch = np.where(idxs_np == bid)[0]
                        loss_in_batch = error_ranks.cpu().numpy()[rel_idx_batch]
                        cur_pose_idx = pose_idx.squeeze().int().numpy()[rel_idx_batch]
                        pid_set = set(cur_pose_idx.tolist())
                        rel_selected_idx = list()
                        for pid in pid_set:
                            pose_mask = cur_pose_idx == pid

                            if self.K_SMALLEST >= len(loss_in_batch[pose_mask]):
                                # Take all the indexes if its not enough.
                                selected_idx = range(len(loss_in_batch[pose_mask]))
                            else:
                                selected_idx = np.argpartition(loss_in_batch[pose_mask].copy(), self.K_SMALLEST)[
                                               :self.K_SMALLEST]
                            rel_selected_idx.extend(rel_idx_batch[pose_mask][selected_idx].tolist())

                        for idx in rel_selected_idx:
                            pred_vertices_render[bid].append(pred_vertices[idx].detach())
                            translation_render[bid].append(translation[idx].detach())
                            bboxes_render[bid].append(pred_bboxes[idx].numpy())
                            head_bboxes_render[bid].append(head_bboxes[idx].numpy())
                            gt_bboxes_render[bid].append(gt_bboxes[idx].numpy())
                            pred_poses2d_render[bid].append(pred_poses2d[idx].numpy())
                            gt_poses2d_render[bid].append(gt_poses2d[idx].numpy())
                            if aten_masks is not None:
                                aten_masks_render[bid].append(aten_masks[idx])
                            if heatmaps is not None:
                                heatmaps_render[bid].append(heatmaps[idx].numpy())

                    raw_images_render = raw_images.clone() * torch.tensor([0.229, 0.224, 0.225],
                                                                          device=raw_images.device) \
                        .view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406], device=raw_images.device).view(1, 3, 1,
                                                                                                               1)
                    pred_vertices_render = [torch.stack(i) for i in pred_vertices_render]
                    translation_render = [torch.stack(i) for i in translation_render]
                    bboxes_render = [np.stack(np.round(i)).astype(int) for i in bboxes_render]
                    head_bboxes_render = [np.stack(np.round(i)).astype(int) for i in head_bboxes_render]
                    gt_bboxes_render = [np.stack(np.round(i)).astype(int) for i in gt_bboxes_render]
                    if aten_masks is not None:
                        aten_masks_render = [np.stack(i) for i in aten_masks_render]
                    if heatmaps is not None:
                        heatmaps_render = [np.stack(i) for i in heatmaps_render]

                    if self.bboxes_only:
                        cur_imgs_rendered = raw_images_render.numpy().copy()
                    else:
                        if self.bird_view:
                            for temp_id in range(raw_images_render.shape[0]):
                                img_t, verts_t, trans_t, bboxes_t = raw_images_render[temp_id], pred_vertices_render[
                                    temp_id], translation_render[temp_id], bboxes_render[temp_id]
                                # import ipdb
                                # ipdb.set_trace()
                                bbox_area = (bboxes_t[:, 2] - bboxes_t[:, 0]) * (bboxes_t[:, 3] - bboxes_t[:, 1])
                                area_mask = torch.tensor(bbox_area > bbox_area.max() * 0.05)
                                verts_t, trans_t = verts_t[area_mask], trans_t[area_mask]
                                verts_t = verts_t + trans_t.unsqueeze(1)
                                verts_tr = torch.einsum('bij,kj->bik', verts_t, R_bv)
                                verts_tfar = verts_tr  # verts_tr + trans_t.unsqueeze(1)
                                p_min, p_max = verts_tfar.view(-1, 3).min(0)[0], verts_tfar.view(-1, 3).max(0)[0]
                                p_center = 0.5 * (p_min + p_max)
                                # trans_tr = torch.einsum('bj,kj->bk', trans_t, R_bv)
                                verts_center = (verts_tfar.view(-1, 3) - p_center).view(verts_t.shape[0], -1, 3)

                                dis_min, dis_max = (verts_tfar.view(-1, 3) - p_center).min(0)[0], (
                                        verts_tfar.view(-1, 3) - p_center).max(0)[0]
                                h, w = img_t.shape[-2:]
                                # h, w = min(h, w), min(h, w)
                                ratio_max = abs(0.9 - 0.5)
                                z_x = dis_max[0] * self.render.focal_length / (ratio_max * w) + torch.abs(dis_min[2])
                                z_y = dis_max[1] * self.render.focal_length / (ratio_max * h) + torch.abs(dis_min[2])
                                z_x_0 = (-dis_min[0]) * self.render.focal_length / (ratio_max * w) + torch.abs(
                                    dis_min[2])
                                z_y_0 = (-dis_min[1]) * self.render.focal_length / (ratio_max * h) + torch.abs(
                                    dis_min[2])
                                z = max(z_x, z_y, z_x_0, z_y_0)
                                verts_right = verts_tfar - p_center + torch.tensor([0, 0, z])
                                img_right = self.render([torch.ones_like(img_t)], [verts_right],
                                                        translation=[torch.zeros_like(trans_t)])
                                bird_view_rendered.extend(img_right)
                                # plt.imshow(to_np(img_right))
                        cur_imgs_rendered = np.stack(
                            self.render(raw_images_render, pred_vertices_render, translation=translation_render))
                    # Add visualization of bboxes on the images.
                    for i, img in enumerate(cur_imgs_rendered):
                        img_cv = (255 * img).transpose([1, 2, 0]).astype(np.uint8).copy()
                        for bbox, pred_pose_vis, gt_pose_vis, head_bbox, gt_bbox in zip(bboxes_render[i],
                                                                                        pred_poses2d_render[i],
                                                                                        gt_poses2d_render[i],
                                                                                        head_bboxes_render[i],
                                                                                        gt_bboxes_render[i]):
                            if self.draw_rpn_bbox:
                                img_cv = cv2.rectangle(img_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                            if self.draw_det_bbox:
                                bbox_color = (0, 0, 255) if gt_pose_vis[..., -1].sum() > 0 else (255, 255, 0)
                                img_cv = cv2.rectangle(img_cv, (head_bbox[0], head_bbox[1]),
                                                       (head_bbox[2], head_bbox[3]), bbox_color, 2)
                            if self.draw_gt_bbox:
                                img_cv = cv2.rectangle(img_cv, (gt_bbox[0], gt_bbox[1]),
                                                       (gt_bbox[2], gt_bbox[3]), (0, 255, 0), 2)
                            if not self.bboxes_only and self.draw_skeleton:  # We don't want to see joints bbox yet.
                                img_cv = draw_skeleton(img_cv, pred_pose_vis[J24_TO_J14])
                        cur_imgs_rendered[i] = np.array(img_cv.transpose([2, 0, 1]) / 255,
                                                        dtype=cur_imgs_rendered.dtype)
                    rendered_imgs.extend(cur_imgs_rendered)

                    if heatmaps is not None:
                        cur_imgs_rendered = raw_images_render.numpy().copy()
                        # Add visualization of bboxes on the images.
                        for i, img in enumerate(cur_imgs_rendered):
                            img_cv = (255 * img).transpose([1, 2, 0]).astype(np.uint8).copy()
                            for bbox, heatmap_vis, gt_pose_vis in zip(bboxes_render[i], heatmaps_render[i],
                                                                      gt_poses2d_render[i]):
                                heatmap_kpts = np.array(
                                    [list(map(lambda x: x[0], np.where(h == h.max()))) for h in heatmap_vis])
                                x1, y1, x2, y2 = bbox
                                w = np.maximum(x2 - x1 + 1, 1)
                                h = np.maximum(y2 - y1 + 1, 1)
                                heatmap_size = heatmap_vis.shape[-1]
                                kpts_img = [y1, x1] + (heatmap_kpts * [h, w]) / heatmap_size
                                kpts_img = kpts_img[:, ::-1]

                                if self.draw_rpn_bbox:
                                    img_cv = cv2.rectangle(img_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0),
                                                           2)
                                if not self.bboxes_only:  # We don't want to see joints bbox yet.
                                    img_cv = draw_skeleton(img_cv, kpts_img[J24_TO_J14], draw_edges=False)
                            cur_imgs_rendered[i] = np.array(img_cv.transpose([2, 0, 1]) / 255,
                                                            dtype=cur_imgs_rendered.dtype)
                        heatmap_imgs.extend(cur_imgs_rendered)

                    if aten_masks is not None:
                        for i, img in enumerate(raw_images_render.numpy()):
                            img_cv = (255 * img).transpose([1, 2, 0]).astype(np.uint8).copy()
                            img_cv = 0.5 * img_cv
                            gap = ((len(bboxes_render[i]) + 1) * 2)
                            num_level = 255 // gap
                            msk_img = np.zeros((img_cv.shape[0], img_cv.shape[1]), dtype=np.uint8)
                            for bidx, (bbox, aten_mask) in enumerate(zip(bboxes_render[i], aten_masks_render[i])):
                                img_cv = cv2.rectangle(img_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                                resized_msk = cv2.resize(aten_mask[0], (bbox[2] - bbox[0], bbox[3] - bbox[1]))
                                msk_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] += (
                                        (resized_msk > self.aten_threshold) * (bidx + 1) * num_level).astype(
                                    np.uint8).copy()
                            mask_bool = np.tile((msk_img == 0)[:, :, np.newaxis], [1, 1, 3])
                            mask_vis = cv2.applyColorMap((msk_img * (gap // 2)).astype(np.uint8), cv2.COLORMAP_RAINBOW)[
                                       :, :, :]
                            mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)
                            mask_vis[mask_bool] = img_cv[mask_bool]
                            img_cv = 0.3 * img_cv + 0.7 * mask_vis
                            aten_mask_imgs.append(
                                np.array(img_cv.transpose([2, 0, 1]) / 255, dtype=cur_imgs_rendered.dtype))
                if self.bboxes_only:
                    total_img = rendered_imgs
                else:
                    total_img = list()
                    for im, bv_im, aten_im, heat_im in zip_longest(rendered_imgs, bird_view_rendered, aten_mask_imgs,
                                                                   heatmap_imgs):
                        total_img.append(im)
                        total_img.append(bv_im)
                        if aten_im is not None:
                            total_img.append(aten_im)
                        if heatmaps is not None:
                            total_img.append(heat_im)
                pred_render_imgs = np.stack(total_img)
                pred_render_imgs[pred_render_imgs < 0] = 0
                pred_render_imgs[pred_render_imgs > 1] = 1
                pred_imgs_vis = torchvision.utils.make_grid(torch.from_numpy(pred_render_imgs), nrow=self.N_ROWS)
                resized_img = self.resizer(pred_imgs_vis.float())
                self.writer.add_image(f'pred_smpl/{runner.mode}', resized_img, runner.iter)
                self.writer.add_histogram(f'cls_dist/{runner.mode}', -loss_in_batch, pseudo_iter)
                self.writer.add_histogram(f'depth/{runner.mode}', translation[:, -1], pseudo_iter)
                self.writer.file_writer.flush()
                # They have some clear operations in base class but I am not sure why the won't work
                # So I clear the log_buffer manually
                for key in runner.log_buffer.output.keys():
                    if key.startswith('img$'):
                        runner.log_buffer.output[key] = list()
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.render.delete()
            # raise e
        self.render.delete()

    @master_only
    def after_run(self, runner):
        self.writer.close()

    def after_train_iter(self, runner):
        # We do not average on the output
        # Just let text hooker do it.s
        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_val_epoch(self, runner):
        # Overwrite to avoid data been averaged again.
        self.log(runner)
        if self.reset_flag:
            runner.log_buffer.clear_output()
