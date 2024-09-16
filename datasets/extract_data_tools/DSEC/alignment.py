# ---------------------------------------------------------------
# Original code from https://github.com/uzh-rpg/DSEC/issues/25.
# ---------------------------------------------------------------

import os
import argparse
from pathlib import Path

import numpy as np
import cv2
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as Rot

class Transform:
    def __init__(self, translation: np.ndarray, rotation: Rot):
        if translation.ndim > 1:
            self._translation = translation.flatten()
        else:
            self._translation = translation
        assert self._translation.size == 3
        self._rotation = rotation

    @staticmethod
    def from_transform_matrix(transform_matrix: np.ndarray):
        translation = transform_matrix[:3, 3]
        rotation = Rot.from_matrix(transform_matrix[:3, :3])
        return Transform(translation, rotation)

    @staticmethod
    def from_rotation(rotation: Rot):
        return Transform(np.zeros(3), rotation)

    def R_matrix(self):
        return self._rotation.as_matrix()

    def R(self):
        return self._rotation

    def t(self):
        return self._translation

    def T_matrix(self) -> np.ndarray:
        return self._T_matrix_from_tR(self._translation, self._rotation.as_matrix())

    def q(self):
        # returns (x, y, z, w)
        return self._rotation.as_quat()

    def euler(self):
        return self._rotation.as_euler('xyz', degrees=True)

    def __matmul__(self, other):
        # a (self), b (other)
        # returns a @ b
        #
        # R_A | t_A   R_B | t_B   R_A @ R_B | R_A @ t_B + t_A
        # --------- @ --------- = ---------------------------
        # 0   | 1     0   | 1     0         | 1
        #
        rotation = self._rotation * other._rotation
        translation = self._rotation.apply(other._translation) + self._translation
        return Transform(translation, rotation)

    def inverse(self):
        #           R_AB  | A_t_AB
        # T_AB =    ------|-------
        #           0     | 1
        #
        # to be converted to
        #
        #           R_BA  | B_t_BA    R_AB.T | -R_AB.T @ A_t_AB
        # T_BA =    ------|------- =  -------|-----------------
        #           0     | 1         0      | 1
        #
        # This is numerically more stable than matrix inversion of T_AB
        rotation = self._rotation.inv()
        translation = - rotation.apply(self._translation)
        return Transform(translation, rotation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqpath', type=str,
                        default='/YourDatasetPath/DESC/desc_seg/test/zurich_city_15_a') # Testing folders: ['zurich_city_13_a', 'zurich_city_14_c', 'zurich_city_15_a']

    args = parser.parse_args()

    seqpath = Path(args.seqpath)
    assert seqpath.is_dir()
    print(f'start processing: {seqpath}')

    confpath = seqpath / 'calibration' / 'cam_to_cam.yaml'
    assert confpath.exists()
    conf = OmegaConf.load(confpath)

    images_left_dir = seqpath / 'images' / 'left'
    outdir = images_left_dir / 'ev_inf'
    os.makedirs(outdir, exist_ok=True)

    image_in_dir = images_left_dir / 'rectified'

    # Get mapping for this sequence:

    K_r0 = np.eye(3)
    K_r0[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['camRect0']['camera_matrix']
    K_r1 = np.eye(3)
    K_r1[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['camRect1']['camera_matrix']

    R_r0_0 = Rot.from_matrix(np.array(conf['extrinsics']['R_rect0']))
    R_r1_1 = Rot.from_matrix(np.array(conf['extrinsics']['R_rect1']))

    T_r0_0 = Transform.from_rotation(R_r0_0)
    T_r1_1 = Transform.from_rotation(R_r1_1)
    T_1_0 = Transform.from_transform_matrix(np.array(conf['extrinsics']['T_10']))

    T_r1_r0 = T_r1_1 @ T_1_0 @ T_r0_0.inverse()
    R_r1_r0_matrix = T_r1_r0.R().as_matrix()
    P_r1_r0 = K_r1 @ R_r1_r0_matrix @ np.linalg.inv(K_r0)

    ht = 480
    wd = 640
    # coords: ht, wd, 2
    coords = np.stack(np.meshgrid(np.arange(wd), np.arange(ht)), axis=-1)
    # coords_hom: ht, wd, 3
    coords_hom = np.concatenate((coords, np.ones((ht, wd, 1))), axis=-1)
    # mapping: ht, wd, 3
    mapping = (P_r1_r0 @ coords_hom[..., None]).squeeze()
    # mapping: ht, wd, 2
    mapping = (mapping/mapping[..., -1][..., None])[..., :2]
    mapping = mapping.astype('float32')

    for entry in image_in_dir.iterdir():
        assert entry.suffix == '.png'
        image_out_file = outdir / entry.name
        if image_out_file.exists():
            continue

        image_in = cv2.imread(str(entry))
        image_out = cv2.remap(image_in, mapping, None, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(image_out_file), image_out)

    print(f'done processing: {seqpath}')
