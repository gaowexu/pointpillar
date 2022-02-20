import torch


class BBoxCoder(object):
    """Bbox Coder for 3D boxes.

    Args:
        code_size (int): The dimension of boxes to be encoded.
    """
    def __init__(self):
        super(BBoxCoder, self).__init__()

    @staticmethod
    def encode(anchor_boxes, gt_boxes):
        """
        对 boxes 进行编码

        :param anchor_boxes: torch.Tensor, 维度为 (N, 7)
        :param gt_boxes: torch.Tensor, 维度为 (N, 7), 需要与 anchor_boxes 维度相同

        :return:
        """
        """Get box regression transformation deltas (dx, dy, dz, dw, dh, dl,
        dr, dv*) that can be used to transform the `src_boxes` into the
        `target_boxes`.

        Args:
            anchor_boxes (torch.Tensor): source boxes, e.g., object proposals.
            gt_boxes (torch.Tensor): target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas.
        """
        xa, ya, za, wa, la, ha, ra = torch.split(anchor_boxes, 1, dim=-1)
        xg, yg, zg, wg, lg, hg, rg = torch.split(gt_boxes, 1, dim=-1)
        za = za + ha / 2
        zg = zg + hg / 2
        diagonal = torch.sqrt(la**2 + wa**2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)
        rt = rg - ra
        return torch.cat([xt, yt, zt, wt, lt, ht, rt], dim=-1)

    @staticmethod
    def decode(anchors, deltas):
        """Apply transformation `deltas` (dx, dy, dz, dw, dh, dl, dr, dv*) to
        `boxes`.

        Args:
            anchors (torch.Tensor): Parameters of anchors with shape (N, 7).
            deltas (torch.Tensor): Encoded boxes with shape
                (N, 7+n) [x, y, z, w, l, h, r, velo*].

        Returns:
            torch.Tensor: Decoded boxes.
        """
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt = torch.split(deltas, 1, dim=-1)

        za = za + ha / 2
        diagonal = torch.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        return torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)


if __name__ == "__main__":
    processor = BBoxCoder()

    import numpy as np
    anchor_boxes = torch.from_numpy(np.array([
        [0, 0, 0, 1.6, 3.9, 1.56, 0.0],
        [10, 10, 10, 1.6, 3.9, 1.56, 0.7853],
    ]))

    gt_boxes = torch.from_numpy(np.array([
        [2.1, 1.0, 1.0, 1.4, 5.2, 1.78, 0.4581],
        [10, 11.2, 13.0, 1.8, 5.0, 1.45, 0.8981],
    ]))

    residuals = processor.encode(anchor_boxes=anchor_boxes, gt_boxes=gt_boxes)
    result = processor.decode(anchors=anchor_boxes, deltas=residuals)
    print("residuals = {}".format(residuals))
    print("result = {}".format(result))



