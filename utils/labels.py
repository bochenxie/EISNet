# ---------------------------------------------------------------
# Original code from https://github.com/uzh-rpg/ess/blob/main/utils/labels.py.
# ---------------------------------------------------------------


import numpy as np
from collections import namedtuple

# a label and all meta information
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])

labels_6_Cityscapes = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 0, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 1, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 1, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 1, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 2, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 2, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 2, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 3, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 3, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 1, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 4, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 4, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 5, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 5, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 5, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 5, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 5, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 5, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
]

Id2label_6_Cityscapes = {label.id: label for label in reversed(labels_6_Cityscapes)}

labels_11_Cityscapes = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 5, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 6, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 1, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 9, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 2, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 4, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 10, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 10, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 7, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 7, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 0, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 3, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 3, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 8, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 8, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 8, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 8, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 8, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 8, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
]

Id2label_11_Cityscapes = {label.id: label for label in reversed(labels_11_Cityscapes)}


def fromIdToTrainId(imgin, Id2label):
    imgout = imgin.copy()
    for id in Id2label:
        imgout[imgin == id] = Id2label[id].trainId
    return imgout


def shiftUpId(imgin):
    imgout = imgin.copy() + 1
    return imgout


def shiftDownId(imgin):
    imgout = imgin.copy()
    imgout[imgin == 0] = 256  # ignore label + 1
    imgout -= 1
    return imgout


def dataset_info(semseg_num_classes):
    # DDD17
    if semseg_num_classes == 6:
        semseg_ignore_label = 255
        semseg_class_names = ['flat', 'background', 'object', 'vegetation', 'human', 'vehicle']
        semseg_color_map = np.zeros((semseg_num_classes, 3), dtype=np.uint8)
        semseg_color_map[0] = [128, 64, 128] # purple
        semseg_color_map[1] = [70, 70, 70] # gray
        semseg_color_map[2] = [220, 220, 0] # yellow
        semseg_color_map[3] = [107, 142, 35] # green
        semseg_color_map[4] = [220, 20, 60] # red
        semseg_color_map[5] = [0, 0, 142] # blue
        return semseg_ignore_label, semseg_class_names, semseg_color_map
    # DESC
    elif semseg_num_classes == 11:
        semseg_ignore_label = 255
        semseg_class_names = ['background', 'building', 'fence', 'person', 'pole', 'road', 'sidewalk', 'vegetation',
                              'car', 'wall', 'traffic sign']
        semseg_color_map = np.zeros((semseg_num_classes, 3), dtype=np.uint8)
        semseg_color_map[0] = [0, 0, 0] # black
        semseg_color_map[1] = [70, 70, 70] # gray
        semseg_color_map[2] = [190, 153, 153] # light red
        semseg_color_map[3] = [220, 20, 60] # red
        semseg_color_map[4] = [153, 153, 153] # gray
        semseg_color_map[5] = [128, 64, 128] # purple
        semseg_color_map[6] = [244, 35, 232] # pink
        semseg_color_map[7] = [107, 142, 35] # green
        semseg_color_map[8] = [0, 0, 142] # dark blue
        semseg_color_map[9] = [102, 102, 156] # gray blue
        semseg_color_map[10] = [220, 220, 0] # yellow
        return semseg_ignore_label, semseg_class_names, semseg_color_map

