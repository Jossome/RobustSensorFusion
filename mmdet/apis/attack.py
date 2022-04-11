import os.path as osp
import pickle
import shutil
import tempfile
import time
import cv2
import skimage.io
import kornia

import mmcv
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmcv.parallel import DataContainer as DC
import numpy as np
import random

from mmdet.core import encode_mask_results, tensor2imgs


def preprocess_yolo(image, for_eval=False, adv_train=False):
    """ Preprocess the data as cfg.data.test does.
    Args:
        image (Variable): image input requires grad
        for_eval: whether to convert image back to mmcv.DataContainer
            for evaluation purpose

    Returns:
        dict: updated `data` dict
    """
    mean=torch.tensor([114, 114, 114, 31.49, 116.71]).float().cuda()
    std=torch.tensor([255, 255, 255, 21.13, 165.34]).float().cuda()
    b, c, h, w = image.shape

    if c == 3:  # RGB image only
        image = image[:, [2, 1, 0], :, :]  # to_rgb=True in this case
        mean = mean[:3]
        std = std[:3]
    elif c == 2:  # depth only
        mean = mean[3:]
        std = std[3:]

    # Normalize
    image = (image - mean[None, :, None, None]) / std[None, :, None, None]

    if for_eval:
        return [image.cpu()]

    return image

def preprocess(image, for_eval=False, adv_train=False):
    """ Preprocess the data as cfg.data.test does.
    Args:
        image (Variable): image input requires grad
        for_eval: whether to convert image back to mmcv.DataContainer
            for evaluation purpose

    Returns:
        dict: updated `data` dict
    """
    mean=torch.tensor([123.675, 116.28, 103.53, 30.30, 119.10]).float().cuda()
    std=torch.tensor([58.395, 57.12, 57.375, 22.18, 167.81]).float().cuda()
    b, c, h, w = image.shape

    if c == 3:  # RGB image only
        image = image[:, [2, 1, 0], :, :]  # to_rgb=True in this case
        mean = mean[:3]
        std = std[:3]

    # Resize:
    if adv_train:
        # TODO implement rand resize and flip
        pass
    else:
        image = F.interpolate(image, size=(1344, 2016))

    # Normalize
    image = (image - mean[None, :, None, None]) / std[None, :, None, None]

    # Pad
    # Seems unnecessary. They just want the size to have 32 as factor

    if for_eval:
        return [image.cpu()]

    return image


def get_metas_yolo(img_metas):
    c = img_metas['ori_shape'][-1]
    if c == 3:
        img_metas['img_norm_cfg'] = {'mean': np.array([114, 114, 114], dtype=np.float32),
                                     'std': np.array([255, 255, 255], dtype=np.float32),
                                     'to_rgb': True}
    elif c == 5:
        img_metas['img_norm_cfg'] = {'mean': np.array([114, 114, 114, 31.49, 116.71], dtype=np.float32),
                                     'std': np.array([255, 255, 255, 21.13, 165.34], dtype=np.float32),
                                     'to_rgb': False}
    elif c == 2:
        img_metas['img_norm_cfg'] = {'mean': np.array([31.49, 116.71], dtype=np.float32),
                                     'std': np.array([21.13, 165.34], dtype=np.float32),
                                     'to_rgb': False}
    else:
        raise NotImplementedError

    return [DC([[img_metas]], cpu_only=True)]


def get_metas(img_metas):
    c = img_metas['ori_shape'][-1]
    img_metas['img_shape'] = (1344, 2016, c)
    img_metas['pad_shape'] = (1344, 2016, c)
    img_metas['scale_factor'] = np.array([1.05, 1.05, 1.05, 1.05], dtype=np.float32)
    if c == 3:
        img_metas['img_norm_cfg'] = {'mean': np.array([123.675, 116.28, 103.53], dtype=np.float32),
                                     'std': np.array([58.395, 57.12, 57.375], dtype=np.float32),
                                     'to_rgb': True}
    elif c == 5:
        img_metas['img_norm_cfg'] = {'mean': np.array([123.675, 116.28, 103.53, 30.30, 119.10], dtype=np.float32),
                                     'std': np.array([58.395, 57.12, 57.375, 22.18, 167.81], dtype=np.float32),
                                     'to_rgb': False}
    else:
        raise NotImplementedError

    return [DC([[img_metas]], cpu_only=True)]


class Attacker:
    def __init__(self, cfg, max_255=False):
        self._build_from_cfg(cfg)
        if max_255:
            self._scale_up_eps()
            self.max_img_value = 255
        else:
            self.max_img_value = 1

    def _build_from_cfg(self, cfg):
        self.method = cfg.method
        self.sensor = cfg.sensor
        self.iters = cfg.iters
        self.alpha_img = cfg.alpha_img
        self.alpha_pts = cfg.alpha_pts
        self.eps_img = cfg.eps_img
        self.eps_pts = cfg.eps_pts
        self.norm = cfg.norm
        self.restrict_region = cfg.restrict_region
        self.patch_size = cfg.patch_size
        self.adv_loss = cfg.adv_loss
        self.random_keep = cfg.random_keep
        if hasattr(cfg, 'mode'):
            self.mode = cfg.mode
        else:
            # Mode: entire; all_car; max_car
            # entire image, all cars, max area car
            self.mode = 'entire'

        if self.alpha_img >= self.eps_img or self.alpha_img * self.iters <= self.eps_img:
            div = 2 if self.iters == 0 else self.iters
            self.alpha_img = self.eps_img / (div / 2)

        if self.alpha_pts >= self.eps_pts or self.alpha_pts * self.iters <= self.eps_pts:
            div = 2 if self.iters == 0 else self.iters
            self.alpha_pts = self.eps_pts / (div / 2)

        if self.alpha_img > 1:
            self.alpha_img /= 255.0
        if self.alpha_pts > 1:
            self.alpha_pts /= 255.0
        if self.eps_img > 1:
            self.eps_img /= 255.0
        if self.eps_pts > 1:
            self.eps_pts /= 255.0

    def _scale_up_eps(self):
        self.alpha_img *= 255
        self.eps_img *= 255

    def _bbox_mask(self, data, mask, largest=False, adv_train=False):
        if adv_train:
            b_gt_bboxes = data['gt_bboxes']
            b_gt_labels = data['gt_labels']

        else:
            b_gt_bboxes = data['gt_bboxes'].data[0]
            b_gt_labels = data['gt_labels'].data[0]

        for gt_bboxes, gt_labels in zip(b_gt_bboxes, b_gt_labels):
            car_bboxes = gt_bboxes[gt_labels == 0]

            if len(car_bboxes) == 0:
                continue

            if largest:
                areas = (car_bboxes[:, 0] - car_bboxes[:, 2]) * (car_bboxes[:, 1] - car_bboxes[:, 3])
                car_bbox = car_bboxes[torch.argmax(areas)]
                x, y, x2, y2 = int(car_bbox[0]), int(car_bbox[1]), int(car_bbox[2]), int(car_bbox[3])
                mask[:, :, y:y2, x:x2] = 1
            else:
                for car_bbox in car_bboxes:
                    x, y, x2, y2 = int(car_bbox[0]), int(car_bbox[1]), int(car_bbox[2]), int(car_bbox[3])
                    mask[:, :, y:y2, x:x2] = 1

    def __call__(self, model, data, adv_train=False):
        assert self.sensor in ['image', 'lidar', 'joint', 'best'], "Unrecognized sensor: '{self.sensor}'"
        return eval(f'self.attack_one_{self.sensor}')(model, data, adv_train=adv_train)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'Attacking {self.sensor} sensor using {self.norm}_{self.method} method.\n'
        if self.sensor in ['image', 'joint']:
            repr_str += f'max_img_value={self.max_img_value}, '
            repr_str += f'eps_img={self.eps_img}, '
            repr_str += f'alpha_img={self.alpha_img}, '
        if self.sensor in ['lidar', 'joint']:
            repr_str += f'eps_pts={self.eps_pts}, '
            repr_str += f'alpha_pts={self.alpha_pts}, '
        repr_str += f'adv_loss={self.adv_loss}, '
        repr_str += f'iters={self.iters}.'
        return repr_str

    def _rfgsm(self, model, data, adv_train=False, within_car=False, largest=False):

        if not adv_train:
            image = data['img'].data[0].float().cuda()
            to_perturb = True
            # Nothing to do with no bbox samples
            if len(data['gt_bboxes'].data[0][0]) == 0:
                to_perturb = False
        else:
            image = data['img']
            to_perturb = random.random() >= self.random_keep

        if to_perturb:
            # Just attack the first three channels.
            rgb_mask = torch.zeros_like(image).float().cuda()
            if within_car:
                self._bbox_mask(data, rgb_mask, largest=largest, adv_train=adv_train)
            else:
                rgb_mask[:, :3, :, :] = 1

            if 'bbox' in self.adv_loss:
                delta_bbox = torch.zeros_like(image).float().uniform_(-self.eps_img, self.eps_img).cuda()
                delta_bbox.requires_grad = True

                perturbed = image + delta_bbox
                perturbed = torch.clamp(perturbed, 0, self.max_img_value)

                data['img'] = preprocess_yolo(perturbed)
                result = model(return_loss=True, **data)

                """ Customize adversarial loss using the results
                loss_cls, loss_bbox, loss_dfl (distributional focal loss)
                """
                sum(result['loss_bbox']).backward()

                delta_bbox = delta_bbox + self.alpha_img * delta_bbox.grad.sign() * rgb_mask
                delta_bbox = torch.clamp(delta_bbox.detach(), -self.eps_img, self.eps_img)

            if 'cls' in self.adv_loss:
                delta_cls = torch.zeros_like(image).float().uniform_(-self.eps_img, self.eps_img).cuda()
                delta_cls.requires_grad = True

                perturbed = image + delta_cls
                perturbed = torch.clamp(perturbed, 0, self.max_img_value)

                data['img'] = preprocess_yolo(perturbed)
                result = model(return_loss=True, **data)

                """ Customize adversarial loss using the results
                loss_cls, loss_cls, loss_dfl (distributional focal loss)
                """
                sum(result['loss_cls']).backward()

                delta_cls = delta_cls + self.alpha_img * delta_cls.grad.sign() * rgb_mask
                delta_cls = torch.clamp(delta_cls.detach(), -self.eps_img, self.eps_img)

            # Select better perturbation:
            if 'cls' in self.adv_loss and 'bbox' in self.adv_loss:
                perturbed = image + delta_cls
                perturbed = torch.clamp(perturbed, 0, self.max_img_value)
                data['img'] = preprocess_yolo(perturbed)
                loss_cls = sum(model(return_loss=True, **data)['loss_cls'])
                perturbed = image + delta_bbox
                perturbed = torch.clamp(perturbed, 0, self.max_img_value)
                data['img'] = preprocess_yolo(perturbed)
                loss_bbox = sum(model(return_loss=True, **data)['loss_bbox'])
                delta = delta_cls if loss_cls > loss_bbox else delta_bbox

            else:
                loss_type = self.adv_loss.split('_')[-1]
                delta = eval(f'delta_{loss_type}')

            new_imgs = torch.clamp(image + delta.detach(), 0, self.max_img_value) * rgb_mask + (1 - rgb_mask) * image

        else:
            new_imgs = image

        new_imgs = preprocess_yolo(new_imgs)
        if adv_train:
            eval_data = data.copy()
            eval_data['img'] = new_imgs
        else:
            eval_data = {'img': [new_imgs],
                         'img_metas': get_metas_yolo(data['img_metas'].data[0][0])}
        return eval_data

    def _rfgsm_lidar(self, model, data, adv_train=False, joint=False, within_car=False, largest=False):

        if not adv_train:
            image = data['img'].data[0].float().cuda()
            points = [data['points'].data[0][0].cuda()]
            to_perturb = True
            # Nothing to do with no bbox samples
            if len(data['gt_bboxes'].data[0][0]) == 0:
                to_perturb = False
        else:
            image = data['img']
            points = data['points']
            to_perturb = random.random() >= self.random_keep

        if to_perturb:
            car_mask = torch.zeros_like(image).float().cuda()
            if within_car:
                self._bbox_mask(data, car_mask, largest=largest, adv_train=adv_train)
                if not joint and image.shape[1] == 5:
                    car_mask[:, :3, :, :] = 0
            else:
                car_mask[:, :, :, :] = 1

            if image.shape[1] == 5:
                rgb_mask = car_mask[:, :3, :, :]
                depth_mask = car_mask[:, 3:, :, :]
            else:
                depth_mask = car_mask

            if within_car:
                data['car_mask'] = depth_mask
                if image.shape[1] == 5:
                    data['depth_map'] = image[:, 3:, :, :]
                else:
                    data['depth_map'] = image

            else:
                data['car_mask'] = None
                data['depth_map'] = None

            if 'bbox' in self.adv_loss:
                delta_bbox = [torch.zeros_like(point).float().uniform_(-self.eps_pts, self.eps_pts).cuda() for point in points]
                for each in delta_bbox:
                    each.requires_grad = True
                data['points'] = [point + each_delta for point, each_delta in zip(points, delta_bbox)]

                if joint:
                    delta_bbox_img = torch.zeros_like(image[:, :3, :, :]).float().uniform_(-self.eps_img, self.eps_img).cuda()
                    delta_bbox_img.requires_grad = True

                if image.shape[1] == 5:
                    if joint:
                        img_input = torch.clamp(image[:, :3, :, :] + delta_bbox_img, 0, self.max_img_value) * rgb_mask + (1 - rgb_mask) * image[:, :3, :, :]
                    else:
                        img_input = image

                else:
                    img_input = image

                data['img'] = img_input
                result = model(return_loss=True, **data)

                """ Customize adversarial loss using the results
                loss_cls, loss_bbox, loss_dfl (distributional focal loss)
                """
                sum(result['loss_bbox']).backward()

                delta_bbox = list(map(lambda x: torch.clamp((x + self.alpha_pts * x.grad.sign()).detach(), -self.eps_pts, self.eps_pts), delta_bbox))

                if joint:
                    delta_bbox_img = delta_bbox_img + self.alpha_img * delta_bbox_img.grad.sign()
                    delta_bbox_img = torch.clamp(delta_bbox_img.detach(), -self.eps_img, self.eps_img)

            if 'cls' in self.adv_loss:
                delta_cls = [torch.zeros_like(point).float().uniform_(-self.eps_pts, self.eps_pts).cuda() for point in points]
                for each in delta_cls:
                    each.requires_grad = True
                data['points'] = [point + each_delta for point, each_delta in zip(points, delta_cls)]

                if joint:
                    delta_cls_img = torch.zeros_like(image[:, :3, :, :]).float().uniform_(-self.eps_img, self.eps_img).cuda()
                    delta_cls_img.requires_grad = True

                if image.shape[1] == 5:
                    if joint:
                        img_input = torch.clamp(image[:, :3, :, :] + delta_cls_img, 0, self.max_img_value)
                    else:
                        img_input = image[:, :3, :, :]

                else:
                    img_input = image

                data['img'] = img_input
                result = model(return_loss=True, **data)

                """ Customize adversarial loss using the results
                loss_cls, loss_cls, loss_dfl (distributional focal loss)
                """
                sum(result['loss_cls']).backward()

                delta_cls = list(map(lambda x: torch.clamp((x + self.alpha_pts * x.grad.sign()).detach(), -self.eps_pts, self.eps_pts), delta_cls))

                if joint:
                    delta_cls_img = delta_cls_img + self.alpha_img * delta_cls_img.grad.sign()
                    delta_cls_img = torch.clamp(delta_cls_img.detach(), -self.eps_img, self.eps_img)

            # Select better perturbation:
            if 'cls' in self.adv_loss and 'bbox' in self.adv_loss:
                data['points'] = [point + each_delta for point, each_delta in zip(points, delta_cls)]
                if image.shape[1] == 5:
                    if joint:
                        img_input = torch.clamp(image[:, :3, :, :] + delta_cls_img, 0, self.max_img_value)
                    else:
                        img_input = image[:, :3, :, :]
                else:
                    img_input = image

                data['img'] = img_input
                loss_cls = sum(model(return_loss=True, **data)['loss_cls'])

                data['points'] = [point + each_delta for point, each_delta in zip(points, delta_bbox)]
                if image.shape[1] == 5:
                    if joint:
                        img_input = torch.clamp(image[:, :3, :, :] + delta_bbox_img, 0, self.max_img_value)
                    else:
                        img_input = image[:, :3, :, :]
                else:
                    img_input = image

                data['img'] = img_input
                loss_bbox = sum(model(return_loss=True, **data)['loss_bbox'])

                delta = delta_cls if loss_cls > loss_bbox else delta_bbox
                if joint:
                    delta_img = delta_cls_img if loss_cls > loss_bbox else delta_bbox_img

            else:
                loss_type = self.adv_loss.split('_')[-1]
                delta = eval(f'delta_{loss_type}')
                if joint:
                    delta_img = eval(f'delta_{loss_type}_img')

            new_pnts = [point + each_delta for point, each_delta in zip(points, delta)]
            if joint:
                new_imgs = torch.clamp(image[:, :3, :, :] + delta_img.detach(), 0, self.max_img_value) * rgb_mask + (1 - rgb_mask) * image[:, :3, :, :]
            else:
                new_imgs = image

        else:
            new_pnts = points
            new_imgs = image
            data['car_mask'] = None
            data['depth_map'] = None

        if adv_train:
            eval_data = data.copy()
            eval_data['img'] = new_imgs
            eval_data['points'] = new_pnts
        else:
            eval_data = {'img': [new_imgs],
                         'points': new_pnts,
                         'img_metas': get_metas_yolo(data['img_metas'].data[0][0])}
        return eval_data

    def _pgd(self, model, data, adv_train=False, within_car=False, largest=False):

        if not adv_train:
            image = data['img'].data[0].float().cuda()
            # Nothing to do with no bbox samples
            to_perturb = True
            if len(data['gt_bboxes'].data[0][0]) == 0:
                to_perturb = False
        else:
            if data['img'].shape[1] == 5:
                depth = data['img'][:, 3:, :, :]
            else:
                depth = None
            image = data['img'][:, :3, :, :]  # For compatibility
            to_perturb = random.random() >= self.random_keep

        delta = torch.zeros_like(image).float().cuda()
        delta.requires_grad = True

        if to_perturb:
            # Just attack the first three channels.
            rgb_mask = torch.zeros_like(image).float().cuda()
            if within_car:
                self._bbox_mask(data, rgb_mask, largest=largest)
            else:
                rgb_mask[:, :3, :, :] = 1

            for i in range(self.iters):
                perturbed = image + delta
                perturbed = torch.clamp(perturbed, 0, self.max_img_value)

                data['img'] = preprocess_yolo(perturbed)
                result = model(return_loss=True, **data)

                """ Customize adversarial loss using the results
                loss_cls, loss_bbox, loss_dfl (distributional focal loss)
                """
                sum(result[self.adv_loss]).backward()

                delta = delta + self.alpha_img * delta.grad.sign() * rgb_mask
                delta = torch.clamp(delta.detach(), -self.eps_img, self.eps_img)
                delta.requires_grad = True

            new_imgs = torch.clamp(image + delta.detach(), 0, self.max_img_value) * rgb_mask + (1 - rgb_mask) * image

        else:
            new_imgs = image

        new_imgs = preprocess_yolo(new_imgs)

        if adv_train:
            eval_data = data.copy()
            if depth is not None:
                eval_data['img'] = torch.cat([new_imgs, depth], axis=1)
            else:
                eval_data['img'] = new_imgs
        else:
            eval_data = {'img': [new_imgs],
                         'img_metas': get_metas_yolo(data['img_metas'].data[0][0])}
        return eval_data

    def _pgd_lidar(self, model, data, adv_train=False, joint=False, best=False, within_car=False, largest=False):

        if not adv_train:
            image = data['img'].data[0].float().cuda()
            points = [data['points'].data[0][0].cuda()]
            # Nothing to do with no bbox samples
            to_perturb = True
            if len(data['gt_bboxes'].data[0][0]) == 0:
                to_perturb = False
        else:
            image = data['img']
            points = data['points']
            to_perturb = random.random() >= self.random_keep

        delta = [torch.zeros_like(point).float().cuda() for point in points]
        for each in delta:
            each.requires_grad = True

        if joint or best:
            delta_img = torch.zeros_like(image[:, :3, :, :]).float().cuda()
            delta_img.requires_grad = True

        if to_perturb:
            car_mask = torch.zeros_like(image).float().cuda()
            if within_car:
                self._bbox_mask(data, car_mask, largest=largest)
                if not joint and image.shape[1] == 5:
                    car_mask[:, :3, :, :] = 0
            else:
                car_mask[:, :, :, :] = 1

            if image.shape[1] == 5:
                rgb_mask = car_mask[:, :3, :, :]
                depth_mask = car_mask[:, 3:, :, :]
            else:
                depth_mask = car_mask

            if within_car:
                data['car_mask'] = depth_mask
                if image.shape[1] == 5:
                    data['depth_map'] = image[:, 3:, :, :]
                else:
                    data['depth_map'] = image

            else:
                data['car_mask'] = None
                data['depth_map'] = None

            for i in range(self.iters):
                data['points'] = [point + each_delta for point, each_delta in zip(points, delta)]
                if image.shape[1] == 5:
                    if joint or best:
                        img_input = torch.clamp(image[:, :3, :, :] + delta_img, 0, self.max_img_value) * rgb_mask + (1 - rgb_mask) * image[:, :3, :, :]
                    else:
                        img_input = image

                else:
                    img_input = image

                data['img'] = img_input
                result = model(return_loss=True, **data)

                """ Customize adversarial loss using the results
                loss_cls, loss_bbox, loss_dfl (distributional focal loss)
                """
                sum(result[self.adv_loss]).backward()

                delta = list(map(lambda x: torch.clamp((x + self.alpha_pts * x.grad.sign()).detach(), -self.eps_pts, self.eps_pts), delta))
                for each in delta:
                    each.requires_grad = True

                if joint or best:
                    delta_img = delta_img + self.alpha_img * delta_img.grad.sign()
                    delta_img = torch.clamp(delta_img.detach(), -self.eps_img, self.eps_img)
                    delta_img.requires_grad = True

            if best:
                data['points'] = points
                data['img'] = torch.clamp(image[:, :3, :, :] + delta_img.detach(), 0, self.max_img_value)
                loss_attack_img = sum(model(return_loss=True, **data))[self.adv_loss]

                data['points'] = [point + each_delta for point, each_delta in zip(points, delta)]
                data['img'] = image
                loss_attack_pts = sum(model(return_loss=True, **data))[self.adv_loss]

                if loss_attack_pts > loss_attack_img:
                    new_pnts = [point + each_delta for point, each_delta in zip(points, delta)]
                    new_imgs = image
                else:
                    new_imgs = torch.clamp(image[:, :3, :, :] + delta_img.detach(), 0, self.max_img_value)
                    new_pnts = points

            else:
                # save_path = 'data/waymococo_f0/' + data['img_metas'].data[0][0]['ori_filename'].replace('jpg', 'pt')
                # torch.save(delta[0], save_path)
                new_pnts = [point + each_delta for point, each_delta in zip(points, delta)]
                if joint:
                    new_imgs = torch.clamp(image[:, :3, :, :] + delta_img.detach(), 0, self.max_img_value) * rgb_mask + (1 - rgb_mask) * image[:, :3, :, :]
                else:
                    new_imgs = image

        else:
            new_pnts = points
            new_imgs = image
            data['car_mask'] = None
            data['depth_map'] = None

        if adv_train:
            eval_data = data.copy()
            eval_data['img'] = new_imgs
            eval_data['points'] = new_pnts
        else:
            eval_data = {'img': [new_imgs],
                         'points': new_pnts,
                         'img_metas': get_metas_yolo(data['img_metas'].data[0][0]),
                         'car_mask': data['car_mask'],
                         'depth_map': data['depth_map']}
        return eval_data

    def attack_one_image(self, model, data, adv_train=False):
        """ Attack one image and return perturbed image (after preprocessing)
        Args:
            model (nn.Module): Model to be attacked
            data (dict): data dictionary containing image input

        Returns:
            dict: data dict with perturbed image
        """
        eval_data = eval(f'self._{self.method}')(model, data, adv_train=adv_train,
                                                 within_car=self.mode != 'entire',
                                                 largest=self.mode == 'max_car')
        return eval_data

    def attack_one_best(self, model, data, adv_train=False):
        """ Attack one image and return perturbed image (after preprocessing)
        Args:
            model (nn.Module): Model to be attacked
            data (dict): data dictionary containing image input

        Returns:
            dict: data dict with perturbed image
        """
        eval_data = eval(f'self._{self.method}_lidar')(model, data, adv_train=adv_train)
        return eval_data

    def attack_one_lidar(self, model, data, adv_train=False):
        """ Attack one lidar and return rgbd image after perturbing and preprocessing
        Args:
            model (nn.Module): Model to be attacked
            data (dict): data dictionary containing image input

        Returns:
            dict: data dict with perturbed image
        """
        eval_data = eval(f'self._{self.method}_lidar')(model, data, adv_train=adv_train,
                                                       within_car=self.mode != 'entire',
                                                       largest=self.mode == 'max_car')
        return eval_data

    def attack_one_joint(self, model, data, adv_train=False):
        """ Attack one lidar and return rgbd image after perturbing and preprocessing
        Args:
            model (nn.Module): Model to be attacked
            data (dict): data dictionary containing image input

        Returns:
            dict: data dict with perturbed image
        """
        eval_data = eval(f'self._{self.method}_lidar')(model, data, adv_train=adv_train, joint=True,
                                                       within_car=self.mode != 'entire',
                                                       largest=self.mode == 'max_car')
        return eval_data



def single_gpu_attack(model,
                      data_loader,
                      show=False,
                      out_dir=None,
                      show_score_thr=0.3,
                      cfg_adv=None,
                      subset=None,
                      ls=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    attacker = Attacker(cfg_adv, max_255=True)

    print(attacker)
    for i, data in enumerate(data_loader):
        # if data['img_metas'].data[0][0]['filename'].split('/')[-1].split('.')[0] != '00056_00090_camera1':
        #     continue

        original = False
        if ls == 'noise':
            data['img']
        else:
            try:
                data = attacker(model, data)
            except:
                original = True

        if ls:
            if attacker.sensor in ['lidar', 'joint']:
                save_path = f'data/waymococo_f0/sub200_adv_images/{i}_d.pt'
                if ls == 'save':
                    adv_depth = data['points']
                    torch.save(adv_depth, save_path)
                else:  # ls == 'load'
                    data['points'] = torch.load(save_path)
            if attacker.sensor in ['image', 'joint']:
                if subset == 'train':
                    if original:
                        fname = data['img_metas'].data[0][0]['filename'].split('/')[-1][:-4]
                    else:
                        fname = data['img_metas'][0].data[0][0]['filename'].split('/')[-1][:-4]
                    # save_path = f'data/waymococo_f0/train_adv_images/{fname}.pt'
                    save_path = f'/cluster-tmp/joss_bbox_job/{fname}.pt'
                else:
                    save_path = f'data/waymococo_f0/sub200_adv_images/{i}.pt'
                if ls == 'save':
                    if original:
                        adv_image = data['img'].data[:, :3, :, :]
                    else:
                        adv_image = data['img'][0][:, :3, :, :]
                    torch.save(adv_image, save_path)
                else:
                    data['img'][0][:, :3, :, :] = torch.load(save_path)

        if subset == 'train':
            prog_bar.update()
            continue
        else:
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)

            batch_size = len(result)

        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, [2, 1, 0]]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    thickness=14,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results
