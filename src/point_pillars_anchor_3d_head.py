import torch
from torch import nn
import numpy as np
from anchors.box_encoder import BBoxCoder
from anchors.anchor_3d_generator import Anchor3DRangeGenerator
from objdet_helper import multiclass_nms, limit_period, bbox_overlaps, box3d_to_bev2d


class PointPillarAnchor3DHead(nn.Module):
    def __init__(self,
                 num_classes=1,
                 in_channels=384,
                 feat_channels=384,
                 nms_pre=100,
                 score_thr=0.1,
                 dir_offset=0,
                 point_cloud_ranges=[[0, -40.0, -3, 70.0, 40.0, 1]],
                 anchor_sizes=[[0.6, 1.0, 1.5]],
                 anchor_rotations=[0, 1.57],
                 iou_thresholds=[[0.35, 0.5]]):

        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.nms_pre = nms_pre
        self.score_thr = score_thr
        self.dir_offset = dir_offset
        self._iou_thresholds = iou_thresholds

        if len(self._iou_thresholds) != num_classes:
            assert len(self._iou_thresholds) == 1
            self._iou_thresholds = self._iou_thresholds * num_classes
        assert len(self._iou_thresholds) == num_classes

        # build anchor generator
        self._anchor_generator = Anchor3DRangeGenerator(
            point_cloud_ranges=point_cloud_ranges,
            anchor_sizes=anchor_sizes,
            anchor_rotations=anchor_rotations
        )
        self._num_anchors_per_location = self._anchor_generator.num_anchors_per_location

        print("self._num_anchors_per_location = {}".format(self._num_anchors_per_location))

        # build box coder
        self._bbox_coder = BBoxCoder()
        self.box_code_size = 7

        self.fp16_enabled = False

        # 构造神经网络预测头
        self.conv_cls = nn.Conv2d(self.feat_channels, self._num_anchors_per_location * self.num_classes, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self._num_anchors_per_location * self.box_code_size, 1)
        self.conv_dir_cls = nn.Conv2d(self.feat_channels, self._num_anchors_per_location * 2, 1)

        self.init_weights()

    @staticmethod
    def bias_init_with_prob(prior_prob):
        """Initialize conv/fc bias value according to giving probablity."""
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
        dir_cls_preds = self.conv_dir_cls(x)

        print("cls_score.shape = {}".format(cls_score.shape))
        print("bbox_pred.shape = {}".format(bbox_pred.shape))
        print("dir_cls_preds.shape = {}".format(dir_cls_preds.shape))
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
        anchors = [
            self._anchor_generator.grid_anchors(pred_bboxes.shape[-2:],
                                               device=pred_bboxes.device)
            for _ in range(len(target_bboxes))
        ]

        # compute size of anchors for each given class
        anchors_cnt = torch.tensor(anchors[0].shape[:-1]).prod()
        rot_angles = anchors[0].shape[-2]

        # init the tensors for the final result
        assigned_bboxes, target_idxs, pos_idxs, neg_idxs = [], [], [], []

        def flatten_idx(idx, j):
            """Inject class dimension in the given indices (...

            z * rot_angles + x) --> (.. z * num_classes * rot_angles + j * rot_angles + x)
            """
            z = idx // rot_angles
            x = idx % rot_angles

            return z * self.num_classes * rot_angles + j * rot_angles + x

        idx_off = 0
        for i in range(len(target_bboxes)):
            for j, (neg_th, pos_th) in enumerate(self._iou_thresholds):
                anchors_stride = anchors[i][..., j, :, :].reshape(
                    -1, self.box_code_size)

                if target_bboxes[i].shape[0] == 0:
                    continue

                # compute a fast approximation of IoU
                overlaps = bbox_overlaps(box3d_to_bev2d(target_bboxes[i]),
                                         box3d_to_bev2d(anchors_stride))

                # for each anchor the gt with max IoU
                max_overlaps, argmax_overlaps = overlaps.max(dim=0)
                # for each gt the anchor with max IoU
                gt_max_overlaps, _ = overlaps.max(dim=1)

                pos_idx = max_overlaps >= pos_th
                neg_idx = (max_overlaps >= 0) & (max_overlaps < neg_th)

                # low-quality matching
                for k in range(len(target_bboxes[i])):
                    if gt_max_overlaps[k] >= neg_th:
                        pos_idx[overlaps[k, :] == gt_max_overlaps[k]] = True

                # encode bbox for positive matches
                assigned_bboxes.append(
                    self._bbox_coder.encode(
                        anchors_stride[pos_idx],
                        target_bboxes[i][argmax_overlaps[pos_idx]]))
                target_idxs.append(argmax_overlaps[pos_idx] + idx_off)

                # store global indices in list
                pos_idx = flatten_idx(
                    pos_idx.nonzero(as_tuple=False).squeeze(-1),
                    j) + i * anchors_cnt
                neg_idx = flatten_idx(
                    neg_idx.nonzero(as_tuple=False).squeeze(-1),
                    j) + i * anchors_cnt
                pos_idxs.append(pos_idx)
                neg_idxs.append(neg_idx)

            # compute offset for index computation
            idx_off += len(target_bboxes[i])

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

        anchors = self._anchor_generator.grid_anchors(cls_scores.shape[-2:],
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

        bboxes = self._bbox_coder.decode(anchors, bbox_preds)

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


if __name__ == "__main__":
    # 参考 https://github.com/open-mmlab/OpenPCDet/blob/master/tools/cfgs/kitti_models/pointpillar.yaml
    dense_head = PointPillarAnchor3DHead(
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        nms_pre=100,
        score_thr=0.1,
        dir_offset=0,
        point_cloud_ranges=[
            [0, -39.68, -1.78, 69.12, 39.68, -1.78],        # 车辆的点云范围
            [0, -39.68, -0.6, 69.12, 39.68, 0.6],           # 行人的点云范围
            [0, -39.68, -0.6, 69.12, 39.68, 0.6],           # 自行车的点云范围
        ],
        anchor_sizes=[
            [1.6, 3.9, 1.56],       # 车辆的anchor尺寸, w, l, h
            [0.6, 0.8, 1.73],       # 行人的anchor尺寸, w, l, h
            [0.6, 1.76, 1.73]       # 自行车的anchor尺寸, w, l, h
        ],
        anchor_rotations=[0, 1.57],
        iou_thresholds=[
            [0.45, 0.60],
            [0.35, 0.50],
            [0.35, 0.50],
        ]
    )

    print(dense_head)

    dense_head(torch.rand((4, 384, 216, 248)))

