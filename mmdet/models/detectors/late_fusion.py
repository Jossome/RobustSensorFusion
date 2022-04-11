import torch
import torch.nn as nn

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


@DETECTORS.register_module()
class LateFusionSSD(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 adv_cfg=None,
                 pretrained=None,
                 fusion='mean'):
        super(LateFusionSSD, self).__init__(adv_cfg)
        backbone['in_channels'] = 3
        self.backbone_rgb = build_backbone(backbone)
        backbone['in_channels'] = 2
        self.backbone_d = build_backbone(backbone)

        assert neck is not None
        self.neck_rgb = build_neck(neck)
        self.neck_d = build_neck(neck)

        self.fusion = fusion
        if 'concat' in self.fusion:
            # Looks like only feat channel is necessary.
            # Not sure why in channel change is optional.
            bbox_head['in_channels'] *= 2
            bbox_head['feat_channels'] *= 2

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(LateFusionSSD, self).init_weights(pretrained)
        self.backbone_rgb.init_weights(pretrained=pretrained)
        self.backbone_d.init_weights(pretrained=pretrained)
        if isinstance(self.neck_rgb, nn.Sequential):
            for m in self.neck_rgb:
                m.init_weights()
        else:
            self.neck_rgb.init_weights()

        if isinstance(self.neck_d, nn.Sequential):
            for m in self.neck_d:
                m.init_weights()
        else:
            self.neck_d.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""

        rgb = img[:, :3, :, :]
        d = img[:, 3:, :, :]

        x_rgb = self.backbone_rgb(rgb)
        x_d = self.backbone_d(d)

        x_rgb = self.neck_rgb(x_rgb)
        x_d = self.neck_d(x_d)

        x = self.fuse_feat(x_rgb, x_d)
        return x

    def _add_feat(self, rgb, d, divisor=1.0):
        # SSD's x is: tuple of length 5, each is a list of length 2.

        # This return is for universenet_latefusion. idk if the yolo version works here.
        # return tuple(torch.stack([(rgb[i][j] + d[i][j]) / divisor for j in range(len(rgb[i]))], dim=0) for i in range(len(rgb)))

        # This return is for yolov4_late
        return tuple((rgb[i] + d[i]) / divisor for i in range(len(rgb)))

    def _concat_feat(self, rgb, d):
        return tuple([torch.cat([rgb[i][j], d[i][j]], dim=1) for j in range(len(rgb[i]))] for i in range(len(rgb)))

    def fuse_feat(self, rgb, d):
        if self.fusion == 'sum':
            return self._add_feat(rgb, d)
        elif self.fusion == 'mean':
            return self._add_feat(rgb, d, divisor=2.0)
        elif self.fusion == 'weighted_mean':
            return self._add_feat(rgb, d, divisor=2.0)
        elif self.fusion == 'concat':
            return self._concat_feat(rgb, d)
        elif self.fusion == 'weighted_concat':
            return self._concat_feat(rgb, d)
        else:
            raise NotImplementedError

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def merge_aug_results(self, aug_bboxes, aug_scores, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).

        Returns:
            tuple: (bboxes, scores)
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                       flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        # recompute feats to save memory
        feats = self.extract_feats(imgs)

        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.bbox_head(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, False)
            det_bboxes, det_scores = self.bbox_head.get_bboxes(*bbox_inputs)[0]
            aug_bboxes.append(det_bboxes)
            aug_scores.append(det_scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_results(
            aug_bboxes, aug_scores, img_metas)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                self.test_cfg.score_thr,
                                                self.test_cfg.nms,
                                                self.test_cfg.max_per_img)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        return [bbox_results]
