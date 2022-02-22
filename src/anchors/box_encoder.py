import torch


class ResidualCoder(object):
    def __init__(self, code_size=7, encode_angle_by_sin_cos=True):
        """
        构建函数

        :param code_size: 三维矩形框的编码维度，默认为7， 代表 x, y, z, dx, dy, dz, heading
        :param encode_angle_by_sin_cos: boolean, 角度编码是否采用正弦、余弦函数编码
        """
        super(ResidualCoder, self).__init__()
        self._code_size = code_size
        self._encode_angle_by_sin_cos = encode_angle_by_sin_cos

        if self._encode_angle_by_sin_cos:
            self._code_size += 1

    @property
    def box_code_size(self):
        return self._code_size

    def encode(self, anchors, boxes):
        """
        基于anchors对输入的boxes进行编码

        :param anchors: 默认三维矩形锚框, torch.Tensor, 形状为 (N, 7)
        :param boxes: 输入的三维矩形框, torch.Tensor, 形状 (N, 7)
        :return:
        """
        anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)

        if self._encode_angle_by_sin_cos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    def decode(self, anchors, deltas):
        """
        基于anchors对预测的残差（residuals）进行解码

        :param anchors: 默认三维矩形锚框, torch.Tensor, 形状为 (batch_size, N, 7+C), 7+C表示
                        [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
        :param deltas: torch.Tensor， 形状为 (batch_size, N, 7+C)
        :return:
        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        if not self._encode_angle_by_sin_cos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(deltas, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(deltas, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza

        if not self._encode_angle_by_sin_cos:
            rg = rt + ra
        else:
            rg_cos = cost + torch.cos(ra)
            rg_sin = sint + torch.sin(ra)
            rg = torch.atan2(rg_sin, rg_cos)

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


if __name__ == "__main__":
    processor = ResidualCoder(encode_angle_by_sin_cos=True)

    import numpy as np
    anchor_boxes = torch.from_numpy(np.array([
        [0, 0, 0, 1.6, 3.9, 1.56, 0.0],
        [10, 10, 10, 1.6, 3.9, 1.56, 0.7853],
    ]))

    gt_boxes = torch.from_numpy(np.array([
        [2.1, 1.0, 1.0, 1.4, 5.2, 1.78, 0.4581],
        [10, 11.2, 13.0, 1.8, 5.0, 1.45, 0.8981]
    ]))

    residuals = processor.encode(anchors=anchor_boxes, boxes=gt_boxes)
    print("residuals = {}".format(residuals))

    result = processor.decode(anchors=anchor_boxes, deltas=residuals)
    print("result = {}".format(result))

