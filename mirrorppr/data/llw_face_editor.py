from __future__ import annotations

from pathlib import Path

from typing import Iterable, Mapping, Sequence

import cv2

import numpy as np

import os

Operation = Mapping[str, object]

class SimilarityMLS:

    def __init__(self, grid_size=50, alpha=1.0):

        self.grid_size = grid_size

        self.alpha = alpha

    def _calculate_weights(self, grid_pts, ctrl_pts):

        n_grid = grid_pts.shape[0]

        n_ctrl = ctrl_pts.shape[0]

        grid_pts_exp = np.tile(grid_pts[:, np.newaxis, :], (1, n_ctrl, 1))

        ctrl_pts_exp = np.tile(ctrl_pts[np.newaxis, :, :], (n_grid, 1, 1))

        d2 = np.sum((grid_pts_exp - ctrl_pts_exp) ** 2, axis=2) + 1e-8

        return 1.0 / (d2 ** self.alpha)

    def warp(self, img, src_pts, dst_pts):

        h, w = img.shape[:2]

        grid_x = np.linspace(0, w, w // self.grid_size + 1)

        grid_y = np.linspace(0, h, h // self.grid_size + 1)

        grid_x, grid_y = np.meshgrid(grid_x, grid_y)

        grid_pts = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

        n_grid = grid_pts.shape[0]

        weights = self._calculate_weights(grid_pts, src_pts)

        total_weights = np.sum(weights, axis=1, keepdims=True)

        p_star = (weights @ src_pts) / total_weights

        q_star = (weights @ dst_pts) / total_weights

        p_hat = src_pts[np.newaxis, :, :] - p_star[:, np.newaxis, :]

        q_hat = dst_pts[np.newaxis, :, :] - q_star[:, np.newaxis, :]

        mu = np.sum(weights[:, :, np.newaxis] * p_hat ** 2, axis=(1, 2), keepdims=True)

        v_hat = grid_pts - p_star

        p_hat_perp = np.stack([-p_hat[:, :, 1], p_hat[:, :, 0]], axis=2)

        vp_dot_phat = np.sum(v_hat[:, np.newaxis, :] * p_hat, axis=2)

        vp_dot_phat_perp = np.sum(v_hat[:, np.newaxis, :] * p_hat_perp, axis=2)

        T1 = vp_dot_phat[:, :, np.newaxis] * q_hat

        T2 = vp_dot_phat_perp[:, :, np.newaxis] * np.stack([-q_hat[:, :, 1], q_hat[:, :, 0]], axis=2)

        weighted_sum = np.sum(weights[:, :, np.newaxis] * (T1 + T2), axis=1)

        result_grid = q_star + weighted_sum / mu.squeeze(-1)

        map_x = cv2.resize(result_grid[:, 0].reshape(grid_x.shape).astype(np.float32), (w, h))

        map_y = cv2.resize(result_grid[:, 1].reshape(grid_y.shape).astype(np.float32), (w, h))

        return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

class FaceEditor:

    def __init__(self):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError("LLW face retouching requires mediapipe. Install the mirrorppr data extra or run the provided environment setup.") from exc

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(

            static_image_mode=True,

            max_num_faces=1,

            refine_landmarks=True,

            min_detection_confidence=0.5

        )

    def _get_landmarks(self, image):

        h, w = image.shape[:2]

        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:

            return None

        landmarks = results.multi_face_landmarks[0].landmark

        points = np.array([(int(l.x * w), int(l.y * h)) for l in landmarks])

        return points

    def _create_roi_mask(self, h, w, points, regions_indices, blur_ratio):

        """
        regions_indices: List[List[int]], 例如 [[左眼索引...], [右眼索引...]]
        """

        mask = np.zeros((h, w), dtype=np.uint8)

        for indices in regions_indices:

            if not indices:

                continue

            roi_points = points[indices].astype(np.int32)

            hull = cv2.convexHull(roi_points)

            cv2.fillConvexPoly(mask, hull, 255)

        blur_k = int(min(h, w) * blur_ratio)

        if blur_k % 2 == 0:

            blur_k += 1

        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

        return mask.astype(np.float32) / 255.0

    def _get_default_params(self, op_type):

        """
        获取操作的默认参数，方便在 process_batch 中合并
        """

        defaults = {

            'grid_size': 30,

            'alpha': 1.0,

            'blur_ratio': 0.05

        }

        return defaults

    def _get_operation_config(self, op_type, landmarks, strength, params):

        """
        根据操作类型分发配置逻辑
        """

        if op_type == 'eye_resize':

            return self._config_eye_resize(landmarks, strength, params)

        elif op_type == 'eye_distance':

            return self._config_eye_distance(landmarks, strength, params)

        elif op_type == 'nose_length':

            return self._config_nose_length(landmarks, strength, params)

        elif op_type == 'nose_alar':

            return self._config_nose_alar(landmarks, strength, params)

        elif op_type == 'mouth_position':

            return self._config_mouth_position(landmarks, strength, params)

        elif op_type == 'lip_thickness':

            return self._config_lip_thickness(landmarks, strength, params)

        elif op_type == 'mouth_resize':

            return self._config_mouth_resize(landmarks, strength, params)

        elif op_type == 'nose_bridge':

            return self._config_nose_bridge(landmarks, strength, params)

        else:

            raise ValueError(f"Unknown operation: {op_type}")

    def _config_nose_length(self, landmarks, strength, params):

        """
        鼻子变短/变长配置
        Strength < 0: 变短 (Shorten) - 鼻头上移，人中变长
        Strength > 0: 变长 (Lengthen) - 鼻头下移，人中变短
        """

        moving_indices = [

            1, 4, 19,

            279, 49,

            2, 98, 327, 456, 236,

            94

        ]

        anchor_idx = [

            168, 6, 197, 195,

            33, 133, 362, 263,

            185,40,39,37,0,267,269,270,409,

            116, 123, 345, 352

        ]

        mask_idx_nose_long = [47, 128, 142, 164, 165, 167, 168, 193, 203, 244, 277, 357, 371, 391, 393, 417, 423, 464]

        mask_groups = [mask_idx_nose_long]

        src_pts = []

        dst_pts = []

        nose_vec_ref = landmarks[1] - landmarks[168]

        nose_length = np.linalg.norm(nose_vec_ref)

        if strength >= 0:

            factor = params.get('max_nose_len_widen_ratio', 0.15)

        else:

            factor = params.get('max_nose_len_shorten_ratio', 0.15)

        move_dist = nose_length * (abs(strength) / 100.0) * factor

        vec_axis = (landmarks[1] - landmarks[168]).astype(np.float32)

        norm_axis = np.linalg.norm(vec_axis)

        if norm_axis > 0: vec_axis /= norm_axis

        if strength >= 0:

            final_vec = vec_axis * move_dist

        else:

            final_vec = -vec_axis * move_dist

        for idx in moving_indices:

            pt = landmarks[idx]

            src_pts.append(pt)

            dst_pts.append(pt + final_vec)

        for idx in anchor_idx:

            pt = landmarks[idx]

            src_pts.append(pt)

            dst_pts.append(pt)

        return np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32), mask_groups

    def _config_nose_alar(self, landmarks, strength, params):

        """
        鼻翼变窄/变宽配置
        Strength < 0: 变窄 (Narrow) - 鼻翼向内收
        Strength > 0: 变宽 (Widen)
        """

        left_alar_indices = [49, 102, 64, 218, 129]

        right_alar_indices = [279, 331, 294, 438, 358]

        anchor_idx = [

            1, 2, 94, 19,

            168, 6, 197, 195, 4,

            0, 37, 267,

            205, 50, 123, 116,

            425, 280, 352, 345

        ]

        mask_idx_alar = [

            195, 4,

            279, 425, 331, 294, 327,

            2, 94,

            98, 64, 102, 205, 49

        ]

        mask_groups = [mask_idx_alar]

        src_pts = []

        dst_pts = []

        alar_width = np.linalg.norm(landmarks[331] - landmarks[102])

        if strength >= 0:

            factor = params.get('max_alar_widen_ratio', 0.25)

            scale = 1.0 + (strength / 100.0) * factor

        else:

            factor = params.get('max_alar_narrow_ratio', 0.25)

            scale = 1.0 - (abs(strength) / 100.0) * factor

        vec_center_line = (landmarks[2] - landmarks[168]).astype(np.float32)

        norm_cl = np.linalg.norm(vec_center_line)

        if norm_cl > 0: vec_center_line /= norm_cl

        def get_projection_point(pt, line_start, line_vec):

            vec_ap = (pt - line_start).astype(np.float32)

            proj_len = np.dot(vec_ap, line_vec)

            return line_start + line_vec * proj_len

        all_moving_indices = left_alar_indices + right_alar_indices

        for idx in all_moving_indices:

            pt = landmarks[idx]

            src_pts.append(pt)

            proj_pt = get_projection_point(pt, landmarks[168], vec_center_line)

            vec_radial = pt - proj_pt

            dst_pts.append(proj_pt + vec_radial * scale)

        for idx in anchor_idx:

            pt = landmarks[idx]

            src_pts.append(pt)

            dst_pts.append(pt)

        return np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32), mask_groups

    def _config_eye_resize(self, landmarks, strength, params):

        """
        眼睛放大/缩小的具体配置
        """

        left_eye_idx = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]

        right_eye_idx = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]

        anchor_idx = [1, 168, 197, 6, 195, 4, 226, 446, 152, 10, 50, 280]

        mask_idx_right =[22, 23, 24, 25, 26, 27, 28, 29, 30, 56, 110, 112, 130, 190, 243, 247]

        mask_idx_left = [252, 253, 254, 255, 256, 257, 258, 259, 260, 286, 339, 341, 359, 414, 463, 467]

        mask_groups = [mask_idx_right, mask_idx_left]

        src_pts = []

        dst_pts = []

        if strength >= 0:

            factor = params.get('max_enlarge', 0.25)

            scale = 1.0 + (strength / 100.0) * factor

        else:

            factor = params.get('max_shrink', 0.20)

            scale = 1.0 + (strength / 100.0) * factor

        def process_region(indices):

            pts = landmarks[indices]

            center = np.mean(pts, axis=0)

            for pt in pts:

                src_pts.append(pt)

                vec = pt - center

                dst_pts.append(center + vec * scale)

        process_region(left_eye_idx)

        process_region(right_eye_idx)

        for idx in anchor_idx:

            pt = landmarks[idx]

            src_pts.append(pt)

            dst_pts.append(pt)

        return np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32), mask_groups

    def _config_eye_distance(self, landmarks, strength, params):

        """
        眼距调整 (修正版：只平移，不缩放)
        Strength > 0: 眼距变宽
        Strength < 0: 眼距变窄
        """

        right_moving_idx = [22, 23, 24, 25, 26, 27, 28, 29, 30, 56, 110, 112, 130, 190, 243, 247]

        left_moving_idx = [252, 253, 254, 255, 256, 257, 258, 259, 260, 286, 339, 341, 359, 414, 463, 467]

        anchor_idx = [

            1, 2, 98, 327,

            10, 152,

            234, 454,

            13, 14, 78, 308

        ]

        mask_idx_right = [31, 113, 189, 221, 222, 223, 224, 225, 226, 228, 229, 230, 231, 232, 233, 244]

        mask_idx_left=[261, 342, 413, 441, 442, 443, 444, 445, 446, 448, 449, 450, 451, 452, 453, 464]

        mask_groups = [mask_idx_right, mask_idx_left]

        src_pts = []

        dst_pts = []

        center_idx = 168

        center_pt = landmarks[center_idx]

        if strength >= 0:

            factor = params.get('max_dist_widen', 0.15)

            scale = 1.0 + (strength / 100.0) * factor

        else:

            factor = params.get('max_dist_narrow', 0.15)

            scale = 1.0 - (abs(strength) / 100.0) * factor

        r_pts = landmarks[right_moving_idx]

        r_centroid = np.mean(r_pts, axis=0)

        r_vec = r_centroid - center_pt

        r_centroid_new = center_pt + r_vec * scale

        r_translation = r_centroid_new - r_centroid

        for idx in right_moving_idx:

            pt = landmarks[idx]

            src_pts.append(pt)

            dst_pts.append(pt + r_translation)

        l_pts = landmarks[left_moving_idx]

        l_centroid = np.mean(l_pts, axis=0)

        l_vec = l_centroid - center_pt

        l_centroid_new = center_pt + l_vec * scale

        l_translation = l_centroid_new - l_centroid

        for idx in left_moving_idx:

            pt = landmarks[idx]

            src_pts.append(pt)

            dst_pts.append(pt + l_translation)

        for idx in anchor_idx:

            pt = landmarks[idx]

            src_pts.append(pt)

            dst_pts.append(pt)

        src_pts.append(center_pt)

        dst_pts.append(center_pt)

        return np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32), mask_groups

    def _config_mouth_position(self, landmarks, strength, params):

        """
        嘴巴上下移动配置 (基于解剖学距离限制)
        Strength < 0: 上移 (最大幅度参照 人中长度)
        Strength > 0: 下移 (最大幅度参照 下唇窝距离)
        """

        lips_indices = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]

        anchor_idx = [

            2, 98, 327, 94, 19, 1,

            152, 377, 148, 365, 136,

            234, 454, 58, 288, 361, 132, 93, 323, 45, 275

        ]

        mask_idx_mouth_zone = [18, 43, 57, 83, 92, 106, 164, 165, 167, 182, 186, 273, 287, 313, 322, 335, 391, 393, 406, 410]

        mask_groups = [mask_idx_mouth_zone]

        src_pts = []

        dst_pts = []

        pt_0 = landmarks[0]

        pt_164 = landmarks[164]

        pt_17 = landmarks[17]

        pt_18 = landmarks[18]

        dist_up_limit = float(abs(pt_0[1] - pt_164[1]))

        dist_down_limit = float(abs(pt_17[1] - pt_18[1]))

        move_y = 0.0

        if strength < 0:

            factor = params.get('limit_mouth_up_factor', 0.8)

            move_dist = dist_up_limit * (abs(strength) / 100.0) * factor

            move_y = -move_dist

        else:

            factor = params.get('limit_mouth_down_factor', 0.8)

            move_dist = dist_down_limit * (strength / 100.0) * factor

            move_y = move_dist

        translation = np.array([0, move_y], dtype=np.float32)

        for idx in lips_indices:

            pt = landmarks[idx]

            src_pts.append(pt)

            dst_pts.append(pt + translation)

        for idx in anchor_idx:

            pt = landmarks[idx]

            src_pts.append(pt)

            dst_pts.append(pt)

        return np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32), mask_groups

    def _config_lip_thickness(self, landmarks, strength, params):

        """
        嘴唇变厚/变薄配置 (严格解剖学版：上下唇独立参照各自厚度)
        """

        upper_lip_outer_moving = [185,40,39,37,0,267,269,270,409]

        lower_lip_outer_moving = [146, 91, 181, 84, 17, 314, 405, 321, 375]

        anchor_idx = [191, 80, 81, 82, 13, 312, 311, 310, 415, 14, 87, 88, 95, 178, 317, 318, 324, 402, 61, 291, 2, 98, 327, 18, 200, 234, 454, 58, 288, 361, 132, 93, 323]

        mask_idx_expanded =[18, 43, 57, 83, 92, 106, 164, 165, 167, 182, 186, 273, 287, 313, 322, 335, 391, 393, 406, 410]

        mask_groups = [mask_idx_expanded]

        src_pts = []

        dst_pts = []

        upper_thickness = np.linalg.norm(landmarks[13] - landmarks[0])

        lower_thickness = np.linalg.norm(landmarks[14] - landmarks[17])

        thickness = max(upper_thickness,lower_thickness)

        if strength >= 0:

            factor = params.get('max_lip_thicken_ratio', 0.4)

        else:

            factor = params.get('max_lip_thin_ratio', 0.3)

        move_dist_upper = thickness * (abs(strength) / 100.0) * factor

        move_dist_lower = thickness * (abs(strength) / 100.0) * factor

        vec_up = (landmarks[2] - landmarks[13]).astype(np.float32)

        norm_up = np.linalg.norm(vec_up)

        if norm_up > 0: vec_up /= norm_up

        vec_down = (landmarks[152] - landmarks[14]).astype(np.float32)

        norm_down = np.linalg.norm(vec_down)

        if norm_down > 0: vec_down /= norm_down

        if strength >= 0:

            final_vec_upper = vec_up * move_dist_upper

            final_vec_lower = vec_down * move_dist_lower

        else:

            final_vec_upper = -vec_up * move_dist_upper

            final_vec_lower = -vec_down * move_dist_lower

        for idx in upper_lip_outer_moving:

            pt = landmarks[idx]

            src_pts.append(pt)

            dst_pts.append(pt + final_vec_upper)

        for idx in lower_lip_outer_moving:

            pt = landmarks[idx]

            src_pts.append(pt)

            dst_pts.append(pt + final_vec_lower)

        for idx in anchor_idx:

            pt = landmarks[idx]

            src_pts.append(pt)

            dst_pts.append(pt)

        return np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32), mask_groups

    def _config_mouth_resize(self, landmarks, strength, params):

        """
        嘴巴整体缩放配置
        Strength > 0: 变大 (Enlarge)
        Strength < 0: 变小 (Shrink)
        """

        lips_indices = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]

        anchor_idx = [

            2, 98, 327,

            205, 425,

            152, 377, 148,

            234, 454, 58, 288, 361, 132, 93, 323,

            164

        ]

        mask_idx_expanded = [18, 43, 57, 83, 92, 106, 164, 165, 167, 182, 186, 273, 287, 313, 322, 335, 391, 393, 406, 410]

        mask_groups = [mask_idx_expanded]

        src_pts = []

        dst_pts = []

        mouth_pts = landmarks[lips_indices]

        center = np.mean(mouth_pts, axis=0)

        if strength >= 0:

            factor = params.get('max_mouth_enlarge', 0.25)

            scale = 1.0 + (strength / 100.0) * factor

        else:

            factor = params.get('max_mouth_shrink', 0.25)

            scale = 1.0 - (abs(strength) / 100.0) * factor

        for idx in lips_indices:

            pt = landmarks[idx]

            src_pts.append(pt)

            vec = pt - center

            dst_pts.append(center + vec * scale)

        for idx in anchor_idx:

            pt = landmarks[idx]

            src_pts.append(pt)

            dst_pts.append(pt)

        src_pts.append(center)

        dst_pts.append(center)

        return np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32), mask_groups

    def _config_nose_bridge(self, landmarks, strength, params):

        """
        鼻梁变窄/变宽配置
        Strength < 0: 变窄 (Narrow) - 更加立体/精致
        Strength > 0: 变宽 (Widen)
        """

        left_bridge_indices = [

            193, 245, 128, 122, 121, 100

        ]

        right_bridge_indices = [

            417, 465, 357, 351, 350, 329

        ]

        anchor_idx =  [

            1, 2, 4, 5, 6, 19, 48, 64, 94, 98, 115, 168, 195, 197,  278, 294, 327, 344,

            362, 133,

            359, 130,

            123, 50, 116,

            352, 280, 345

        ]

        mask_idx_bridge = [9, 55, 97, 134, 164, 174, 188, 193, 220, 236, 237, 245, 285, 326, 363, 399, 412, 417, 440, 456, 457, 465]

        mask_groups = [mask_idx_bridge]

        src_pts = []

        dst_pts = []

        if strength >= 0:

            factor = params.get('max_nose_widen_ratio', 0.3)

            scale = 1.0 + (strength / 100.0) * factor

        else:

            factor = params.get('max_nose_narrow_ratio', 0.25)

            scale = 1.0 - (abs(strength) / 100.0) * factor

        vec_center_line = (landmarks[2] - landmarks[168]).astype(np.float32)

        norm_cl = np.linalg.norm(vec_center_line)

        if norm_cl > 0: vec_center_line /= norm_cl

        def get_projection_point(pt, line_start, line_vec):

            vec_ap = (pt - line_start).astype(np.float32)

            proj_len = np.dot(vec_ap, line_vec)

            return line_start + line_vec * proj_len

        all_moving_indices = [3, 44, 45, 51, 122, 193, 196, 248, 274, 275, 281, 351, 417, 419]

        for idx in all_moving_indices:

            pt = landmarks[idx]

            src_pts.append(pt)

            proj_pt = get_projection_point(pt, landmarks[168], vec_center_line)

            vec_radial = pt - proj_pt

            dst_pts.append(proj_pt + vec_radial * scale)

        for idx in anchor_idx:

            pt = landmarks[idx]

            src_pts.append(pt)

            dst_pts.append(pt)

        return np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32), mask_groups

    def apply_deformation(self, image, src_pts, dst_pts, mask_indices, landmarks, params):

        h, w = image.shape[:2]

        mls = SimilarityMLS(

            grid_size=params.get('grid_size', 50),

            alpha=params.get('alpha', 1.0)

        )

        warped_image = mls.warp(image, dst_pts, src_pts)

        mask = self._create_roi_mask(

            h, w, landmarks, mask_indices,

            blur_ratio=params.get('blur_ratio', 0.08)

        )

        mask = mask[:, :, np.newaxis]

        result = warped_image * mask + image * (1.0 - mask)

        return result.astype(np.uint8)

    def process_and_save(self, image_path, output_path, op_type='eye_resize', strength=50, hyperparams=None):

        """
        单次操作的便捷入口，内部调用 process_batch
        """

        operation = {

            'op_type': op_type,

            'strength': strength,

            'params': hyperparams if hyperparams else {}

        }

        self.process_batch(image_path, output_path, [operation])

    def process_batch(self, image_path, output_path, operations, diff_output_path=None):

        """
        批量执行多个编辑操作 (Pipeline模式)
        :param operations: 操作列表
        :param diff_output_path: (新增) 指定差异图的保存路径，如果为None则不保存或使用默认命名
        """

        img = cv2.imread(image_path)

        if img is None:

            print(f"Error: Could not read image: {image_path}")

            return

        arr_source = img.astype(np.float32)

        current_img = img.copy()

        for i, op in enumerate(operations):

            op_type = op.get('op_type')

            strength = op.get('strength')

            custom_params = op.get('params', {})

            landmarks = self._get_landmarks(current_img)

            if landmarks is None:

                break

            params = self._get_default_params(op_type)

            if custom_params:

                params.update(custom_params)

            try:

                src_pts, dst_pts, mask_groups = self._get_operation_config(

                    op_type, landmarks, strength, params

                )

                current_img = self.apply_deformation(

                    current_img, src_pts, dst_pts, mask_groups, landmarks, params

                )

            except ValueError as e:

                continue

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cv2.imwrite(output_path, current_img)

class LLWFaceRetoucher(FaceEditor):

    """Public wrapper for Landmark-Guided Local Warping face retouching."""

    def apply_operations_to_array(self, image_bgr: np.ndarray, operations: Sequence[Operation]) -> np.ndarray:

        current_img = image_bgr.copy()

        for op in operations:

            op_type = str(op.get("op_type") or op.get("operation") or op.get("name"))

            strength = float(op.get("strength", op.get("value", 100)))

            custom_params = op.get("params", {}) or {}

            landmarks = self._get_landmarks(current_img)

            if landmarks is None:

                raise RuntimeError("No face landmarks were detected before applying operation: " + op_type)

            params = self._get_default_params(op_type)

            params.update(custom_params)

            src_pts, dst_pts, mask_groups = self._get_operation_config(op_type, landmarks, strength, params)

            current_img = self.apply_deformation(current_img, src_pts, dst_pts, mask_groups, landmarks, params)

        return current_img

    def apply_operations(self, image_path: str | Path, operations: Sequence[Operation], output_path: str | Path | None = None) -> np.ndarray:

        image = cv2.imread(str(image_path))

        if image is None:

            raise FileNotFoundError(f"Could not read image: {image_path}")

        result = self.apply_operations_to_array(image, operations)

        if output_path is not None:

            output_path = Path(output_path)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(output_path), result)

        return result

def normalize_operations(operation_specs: Iterable[str]) -> list[dict[str, object]]:

    operations = []

    for spec in operation_specs:

        if ":" not in spec:

            raise ValueError(f"Operation must be formatted as name:strength, got {spec!r}")

        name, value = spec.split(":", 1)

        operations.append({"op_type": name.strip(), "strength": float(value)})

    return operations
