import torch
from torch import nn
import numpy as np
from objdet_helper import Anchor3DRangeGenerator, BBoxCoder, multiclass_nms, limit_period, get_paddings_indicator, bbox_overlaps, box3d_to_bev2d


class PointPillarAnchor3DHead(nn.Module):
    def __init__(self,
                 num_classes=1,
                 in_channels=384,
                 feat_channels=384,
                 nms_pre=100,
                 score_thr=0.1,
                 dir_offset=0,
                 ranges=[[0, -40.0, -3, 70.0, 40.0, 1]],
                 sizes=[[0.6, 1.0, 1.5]],
                 rotations=[0, 1.57],
                 iou_thr=[[0.35, 0.5]]):
        super().__init__()
        self._num_classes = num_classes
        self._in_channels = in_channels
        self._feat_channels = feat_channels
        self._nms_pre = nms_pre
        self._score_thr = score_thr
        self._dir_offset = dir_offset
        self._ranges = ranges
        self._sizes = sizes
        self._rotations = rotations
        self._iou_thr = iou_thr



        if len(self.iou_thr) != num_classes:
            assert len(self.iou_thr) == 1
            self.iou_thr = self.iou_thr * num_classes
        assert len(self.iou_thr) == num_classes

        # build anchor generator
        self.anchor_generator = Anchor3DRangeGenerator(ranges=ranges,
                                                       sizes=sizes,
                                                       rotations=rotations)
        self.num_anchors = self.anchor_generator.num_base_anchors

        # build box coder
        self.bbox_coder = BBoxCoder()
        self.box_code_size = 7

        self.fp16_enabled = False

        #Initialize neural network layers of the head.
        self.cls_out_channels = self.num_anchors * self.num_classes
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.box_code_size, 1)
        self.conv_dir_cls = nn.Conv2d(self.feat_channels, self.num_anchors * 2,
                                      1)

        self.init_weights()

    @staticmethod
    def bias_init_with_prob(prior_prob):
        """initialize conv/fc bias value according to giving probablity."""
        bias_init = float(-np.log((1 - prior_prob) / prior_prob))

        return bias_init

    @staticmethod
    def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def init_weights(self):
        """Initialize the weights of head."""
        bias_cls = self.bias_init_with_prob(0.01)
        self.normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        self.normal_init(self.conv_reg, std=0.01)

    def forward(self, x):
        """Forward function on a feature map.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            tuple[torch.Tensor]: Contain score of each class, bbox \
                regression and direction classification predictions.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_preds = None
        dir_cls_preds = self.conv_dir_cls(x)
        return cls_score, bbox_pred, dir_cls_preds

    def assign_bboxes(self, pred_bboxes, target_bboxes):
        """Assigns target bboxes to given anchors.

        Args:
            pred_bboxes (torch.Tensor): Bbox predictions (anchors).
            target_bboxes (torch.Tensor): Bbox targets.

        Returns:
            torch.Tensor: Assigned target bboxes for each given anchor.
            torch.Tensor: Flat index of matched targets.
            torch.Tensor: Index of positive matches.
            torch.Tensor: Index of negative matches.
        """

        # compute all anchors
        anchors = self.anchor_generator.grid_anchors(pred_bboxes.shape[-2:],
                                                     device=pred_bboxes.device)

        rot_angles = anchors[0].shape[-2]

        # init the tensors for the final result
        assigned_bboxes, target_idxs, pos_idxs, neg_idxs = [], [], [], []

        def flatten_idx(idx, j):
            """inject class dimension in the given indices (... z * rot_angles + x) --> (.. z * num_classes * rot_angles + j * rot_angles + x)"""
            z = idx // rot_angles
            x = idx % rot_angles

            return z * self.num_classes * rot_angles + j * rot_angles + x

        for i, (neg_th, pos_th) in enumerate(self.iou_thr):
            anchors_stride = anchors[...,
                                     i, :, :].reshape(-1, self.box_code_size)

            # compute a fast approximation of IoU
            overlaps = bbox_overlaps(box3d_to_bev2d(target_bboxes),
                                     box3d_to_bev2d(anchors_stride))

            # for each anchor the gt with max IoU
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)
            # for each gt the anchor with max IoU
            gt_max_overlaps = overlaps.max(dim=1)[0]

            pos_idx = max_overlaps >= pos_th
            neg_idx = (max_overlaps >= 0) & (max_overlaps < neg_th)

            # low-quality matching
            for k in range(len(target_bboxes)):
                if gt_max_overlaps[k] >= neg_th:
                    pos_idx[overlaps[k, :] == gt_max_overlaps[k]] = True

            # encode bbox for positive matches
            assigned_bboxes.append(
                self.bbox_coder.encode(anchors_stride[pos_idx],
                                       target_bboxes[argmax_overlaps[pos_idx]]))
            target_idxs.append(argmax_overlaps[pos_idx])

            # store global indices in list
            pos_idx = flatten_idx(
                pos_idx.nonzero(as_tuple=False).squeeze(-1), i)
            neg_idx = flatten_idx(
                neg_idx.nonzero(as_tuple=False).squeeze(-1), i)
            pos_idxs.append(pos_idx)
            neg_idxs.append(neg_idx)

        return (torch.cat(assigned_bboxes,
                          axis=0), torch.cat(target_idxs, axis=0),
                torch.cat(pos_idxs, axis=0), torch.cat(neg_idxs, axis=0))

    def get_bboxes(self, cls_scores, bbox_preds, dir_preds):
        """Get bboxes of anchor head.

        Args:
            cls_scores (list[torch.Tensor]): Class scores.
            bbox_preds (list[torch.Tensor]): Bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Direction
                class predictions.

        Returns:
            tuple[torch.Tensor]: Prediction results of batches
                (bboxes, scores, labels).
        """
        bboxes, scores, labels = [], [], []
        for cls_score, bbox_pred, dir_pred in zip(cls_scores, bbox_preds,
                                                  dir_preds):
            b, s, l = self.get_bboxes_single(cls_score, bbox_pred, dir_pred)
            bboxes.append(b)
            scores.append(s)
            labels.append(l)
        return bboxes, scores, labels

    def get_bboxes_single(self, cls_scores, bbox_preds, dir_preds):
        """Get bboxes of anchor head.

        Args:
            cls_scores (list[torch.Tensor]): Class scores.
            bbox_preds (list[torch.Tensor]): Bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Direction
                class predictions.

        Returns:
            tuple[torch.Tensor]: Prediction results of batches
                (bboxes, scores, labels).
        """
        assert cls_scores.size()[-2:] == bbox_preds.size()[-2:]
        assert cls_scores.size()[-2:] == dir_preds.size()[-2:]

        anchors = self.anchor_generator.grid_anchors(cls_scores.shape[-2:],
                                                     device=cls_scores.device)
        anchors = anchors.reshape(-1, self.box_code_size)

        dir_preds = dir_preds.permute(1, 2, 0).reshape(-1, 2)
        dir_scores = torch.max(dir_preds, dim=-1)[1]

        cls_scores = cls_scores.permute(1, 2, 0).reshape(-1, self.num_classes)
        scores = cls_scores.sigmoid()

        bbox_preds = bbox_preds.permute(1, 2, 0).reshape(-1, self.box_code_size)

        if scores.shape[0] > self.nms_pre:
            max_scores, _ = scores.max(dim=1)
            _, topk_inds = max_scores.topk(self.nms_pre)
            anchors = anchors[topk_inds, :]
            bbox_preds = bbox_preds[topk_inds, :]
            scores = scores[topk_inds, :]
            dir_scores = dir_scores[topk_inds]

        bboxes = self.bbox_coder.decode(anchors, bbox_preds)

        idxs = multiclass_nms(bboxes, scores, self.score_thr)

        labels = [
            torch.full((len(idxs[i]),), i, dtype=torch.long)
            for i in range(self.num_classes)
        ]
        labels = torch.cat(labels)

        scores = [scores[idxs[i], i] for i in range(self.num_classes)]
        scores = torch.cat(scores)

        idxs = torch.cat(idxs)
        bboxes = bboxes[idxs]
        dir_scores = dir_scores[idxs]

        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset, 1, np.pi)
            bboxes[..., 6] = (dir_rot + self.dir_offset +
                              np.pi * dir_scores.to(bboxes.dtype))

        return bboxes, scores, labels


















