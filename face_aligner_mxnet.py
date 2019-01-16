#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yongjie shi, zhaoyafei
import _init_paths

import os
import os.path as osp

import cv2
import math
import numpy as np

from mxnet_feature_extractor import MxnetFeatureExtractor
from fx_warp_and_crop_face import warp_and_crop_face, get_reference_facial_points


def convert_to_squares(pts, scale=1.0):
    # convert pts to square
    x, y = pts[0], pts[1]
    w, h = pts[2], pts[3]
    max_side = np.maximum(w, h) * scale
    x += w * 0.5 - max_side * 0.5
    y += h * 0.5 - max_side * 0.5
    if x < 1:
        x = 1
    if y < 1:
        y = 1
    return int(x), int(y), int(max_side)


def get_roi_img(img, pts):
    """Get ROI region from input image. Part of the ROI can be outside of the image.

    Params:
        img: input image, numpy array.
        pts: top-left and bottom-right points of ROI, [[x1, y1], [x2, y2]]
    Return:
        ROI image, numpy array
    """
    x1 = pts[0][0]
    y1 = pts[0][1]
    x2 = pts[1][0]
    y2 = pts[1][1]
    w, h = x2-x1, y2-y1

    img_shape = img.shape
    roi_shape = (h, w)
    if len(img_shape) > 2:
        roi_shape = (h, w, img_shape[2])

    img_roi = np.zeros(roi_shape, dtype=img.dtype)

    dst_x1 = 0
    dst_y1 = 0
    src_x1 = x1
    src_y1 = y1

    if x1 < 0:
        dst_x1 = -x1-1
        w += x1
        src_x1 -= x1

    if y1 < 0:
        dst_y1 = -y1-1
        h += y1
        src_y1 -= y1

    if x2 > img_shape[1] - 1:
        w -= x2 - img_shape[1] + 1

    if y2 > img_shape[0] - 1:
        h -= y2 - img_shape[0] + 1

    img_roi[dst_y1: dst_y1+h, dst_x1: dst_x1 +
            w] = img[src_y1:src_y1+h, src_x1:src_x1+w]

    return img_roi


def get_center_roi(img, scale):
    """Get ROI region from input image. Part of the ROI can be outside of the image.

    Params:
        img: input image, numpy array.
        scale: ratio of center roi size to image size
    Return:
        ROI image, numpy array
    """
    img_shape = img.shape
    h, w = img_shape[0], img_shape[1]

    new_h = int(h * scale)
    new_w = int(w * scale)

    x1 = (h-new_h) // 2
    y1 = (w-new_w) // 2

    pts = [[x1, y1], [x1+new_w, y1+new_h]]

    roi_img = get_roi_img(img, pts)

    return roi_img


def rotate_point(x, y, centerX, centerY, angle):
    """Rotate point by angle around center.

    Params:
        x,y: input point
        centerX, centerY: center
        angle: float, in degree [-180, 180] not radian
    Return:
        rotated point, [rx, ry]
    """
    if angle < 1:  # skip angle < 1 degree
        return x, y

    angle = angle * math.pi / 180
    x -= centerX
    y -= centerY
    theta = -angle  # * math.pi / 180

    rx = (int)(centerX + x * math.cos(theta) - y * math.sin(theta))
    ry = (int)(centerY + x * math.sin(theta) + y * math.cos(theta))

    return [rx, ry]


def get_upright_face(img, pts, angle, scale=1.0):
    """Rotate face to upright position.

    Params:
        img: input image, numpy array
        pts: face rect, 4 pts, [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        angle: float, in degree [-180, 180] not radian
        scale: output size = scale * sqrt(w*w+h*h), w,h is the size of ROI
    Return:
        rotated and cropped upright face image, numpy array
    """
    x1, y1 = pts[0][0], pts[0][1]
    x2, y2 = pts[2][0]-1, pts[2][1]-1
    w, h = x2-x1, y2-y1

    w_max = int(math.sqrt(w*w+h*h))
    crop_size = int(w_max*scale)
    half_crop_size = crop_size/2

    center_x = (x1+w/2)
    center_y = (y1+h/2)

    if angle < 1.0:  # skip angle < 1 degree
        new_pts = [
            [center_x - half_crop_size, center_y - half_crop_size],
            [center_x + half_crop_size, center_y + half_crop_size]
        ]
        face_img = get_roi_img(img, new_pts)
    else:
        src1 = rotate_point(center_x-half_crop_size, center_y -
                            half_crop_size, center_x, center_y, angle)
        src2 = rotate_point(center_x-half_crop_size, center_y +
                            half_crop_size, center_x, center_y, angle)
        src3 = rotate_point(center_x+half_crop_size, center_y +
                            half_crop_size, center_x, center_y, angle)

        dst1 = [0, 0]
        dst2 = [0, crop_size]
        dst3 = [crop_size, crop_size]

        pts1 = np.float32([src1, src2, src3])
        pts2 = np.float32([dst1, dst2, dst3])

        M = cv2.getAffineTransform(pts1, pts2)
        face_img = cv2.warpAffine(img, M, (crop_size, crop_size))

    return face_img


def mark_img_with_pts(im, pts):
    """draw landmarks onto image.

    """
    for pt in pts:
        cv2.circle(im, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), 3)
    return im


class FaceAlignerCaffe(object):
    """Face Aligner using only one network.

    """

    def __init__(self, config_json):
        """Face Aligner using only one network.

            Params:
                config_json: config params for the inference network
        """
        self.net_handle = MxnetFeatureExtractor(config_json)
        self.batch_size = self.net_handle.get_batch_size()
        self.net_input_width = config_json["input_width"]
        self.net_input_height = config_json["input_height"]

        self.feature_layers = self.net_handle.feature_layers
        self.net_output_layer = self.net_handle.get_feature_layers()[0]

    def get_landmarks(self, im_list, center_roi_scale=1.0):
        """Get landmarks for every image in a image list.

        Params:
            img: a list of images, each one is a numpy array
            center_roi_scale: only use center roi to do net inference
        Return:
            a list of face landmarks, has the same length of input im_list
        """
        five_pts_list = []
        size = len(im_list)

        im_list2 = []
        if center_roi_scale > 1.0:
            raise Exception("scale must be <= 1.0")
        elif center_roi_scale < 0.99:
            for i in range(size):
                center_roi = get_center_roi(im_list[i], center_roi_scale)
                im_list2.append(center_roi)
        else:
            im_list2 = im_list

        for k in range(0, size, self.batch_size):
            infer_batch = self.batch_size
            if k + self.batch_size > size:
                infer_batch = size - k

            infer_res = self.net_handle.extract_features_batch(
                im_list2[k:k + infer_batch])

            for j in range(infer_batch):
                five_pts = infer_res[self.net_output_layer][j]
                # five_pts = infer_res[j]

                img_shape = im_list2[k + j].shape
                # print('---> image shape: ', img_shape)
                img_ht = img_shape[0]
                img_wd = img_shape[1]

                five_pts = np.reshape(five_pts, (2, -1)).T
                five_pts[:, 1] = five_pts[:, 1] * img_ht
                five_pts[:, 0] = five_pts[:, 0] * img_wd
                # five_pts[:, 0] = five_pts[:, 0] * img_wd / self.net_input_width
                # five_pts[:, 1] = five_pts[:, 1] * img_ht / self.net_input_height
                # print("---> 1 pts: ", five_pts)

                if center_roi_scale < 0.99:  # add offset for center ROI
                    offset_x = img_wd * \
                        (1.0-center_roi_scale) / center_roi_scale * 0.5
                    offset_y = img_ht * \
                        (1.0-center_roi_scale) / center_roi_scale * 0.5
                    five_pts[:, 0] += offset_x
                    five_pts[:, 1] += offset_y

                    # print("---> 2 pts: ", five_pts)

                five_pts_list.append(five_pts)

        return five_pts_list

    # pts_with_angles list of [[[1,2],[3,4],[5,6],[7,8]],1(angle)]
    def rotate_and_crop_faces(self, img, pts_with_angles, scale=1.0):
        """Rotate face rects into upright position and crop them out.

        Params:
            img: input image, numpy array
            pts_with_angles: a list of (pts, angle) pairs,
                    pts: face rect, 4 pts, [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    angle: float, in degree [-180, 180] not radian
            scale: output size = scale * sqrt(w*w+h*h), w,h is the size of ROI
                    use scale>1.0 (i.e. 1.5) to avoid "black triangles" when doing face alignment
        Return:
            a list of rotated and cropped face roi images (eacho one is a numpy array),
            the output list has the same length of input pts_with_angles
        """
        img_cropped_list = []

        for pt_angle in pts_with_angles:
            pts, angle = pt_angle[0], pt_angle[1]
            print('pts={}'.format(pts))
            print('angle={}'.format(angle))

            if not isinstance(angle, float):
                angle = (float)(angle)
            img_cropped = get_upright_face(img, pts, angle, scale)
            img_cropped_list.append(img_cropped)

        return img_cropped_list

    def get_aligned_face_chips(self, img_list, facial_points_list, output_square=True):
        """Get aligned face chips in a image list.

        Params:
            img_list: a list of input images, each image is a numpy array
            facial_points_list: a list of face landmarks, has the same length as img_list, each one is for one image
            output_square: whether to output square face chips
        Return:
            a list of aligned face roi chips (eacho one is a numpy array),
            the output list has the same length of input pts_with_angles
        """
        face_chips = []
        output_size = (96, 112)  # (w, h) not (h,w)

        if output_square:
            output_size = (112, 112)
            reference_5pts = get_reference_facial_points(output_size)

        for img, facial_points in zip(img_list, facial_points_list):
            facial_5pts = np.reshape(facial_points, (5, -1))
            dst_img = warp_and_crop_face(
                img, facial_5pts, reference_5pts, output_size)
            face_chips.append(dst_img)

        return face_chips


if __name__ == '__main__':
    import json

    save_res_imgs = True
    test_file = 'test_imgs_weidong/test_data.json'
    config_file = 'face_aligner_config.json'
    save_dir = 'rlt_images'
    if not osp.exists(save_dir):
        os.mkdir(save_dir)

    with open(config_file, 'r') as fout:
        json_str = json.load(fout)

    face_aligner = FaceAlignerCaffe(json_str)

    with open(test_file, 'r') as fout:
        json_str = json.load(fout)
      #  total_pts_list = []
        total_img_cropped_list = []
        aligned_faces_list = []
        five_pts_list = []

        for idx, body in enumerate(json_str):
            pts_with_angles = []
            img_cropped_list = []
            if 'uri' in body:
                uri = body["uri"]

            if 'detections' in body:
                data_list = body['detections']
                for data in data_list:
                    if 'pts' in data:
                        pts = data['pts']
                    if 'quality' in data and data['quality'] == 'small':
                        continue
                    if 'orientation' in data:
                        angle = data['orientation']
                    # convert radians into degrees
                    pts_with_angles.append([pts, angle*180/math.pi])

                size = len(pts_with_angles)

                print('uri={}'.format(uri))
                if not size:
                    print("No faces found")
                    continue

                img = cv2.imread(uri)

                img_cropped_list = face_aligner.rotate_and_crop_faces(
                    img, pts_with_angles, scale=1.5)  # use scale>1.0 to avoid "black triangles" in face chips
                total_img_cropped_list.extend(img_cropped_list)

            print('total_img_cropped_list size={}'.format(
                len(total_img_cropped_list)))

        if save_res_imgs:
            sub_dir = save_dir + '/cropped'
            if not osp.exists(sub_dir):
                os.mkdir(sub_dir)

            for idx, img_cropped in enumerate(total_img_cropped_list):
                file_name = str(idx+1)+'.jpg'
                file_name = osp.join(sub_dir, file_name)
                cv2.imwrite(file_name, img_cropped)

        five_pts_list = face_aligner.get_landmarks(
            total_img_cropped_list, 1/1.5*0.9)

        if save_res_imgs:
            sub_dir = save_dir + '/cropped_with_landmarks'

            if not osp.exists(sub_dir):
                os.mkdir(sub_dir)

            for idx, img_cropped in enumerate(total_img_cropped_list):
                five_pts = five_pts_list[idx]
                print('---> five_pts={}'.format(five_pts))
                mark_img_with_pts(img_cropped, five_pts)

                file_name = str(idx+1)+'.jpg'
                file_name = osp.join(sub_dir, file_name)
                cv2.imwrite(file_name, img_cropped)

        aligned_faces_list = face_aligner.get_aligned_face_chips(
            total_img_cropped_list, five_pts_list)

        if save_res_imgs:
            sub_dir = save_dir + '/aligned_faces'
            if not osp.exists(sub_dir):
                os.mkdir(sub_dir)

            for idx, face_chip in enumerate(aligned_faces_list):
                file_name = str(idx+1)+'.jpg'
                file_name = osp.join(sub_dir, file_name)
                cv2.imwrite(file_name, face_chip)
