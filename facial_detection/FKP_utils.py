import random
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import FKP_opt


class Normalize(object):
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        return {'image': image / 255., 'keypoints': keypoints}


class ToTensor(object):
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        # swap color axis because
        image = image.reshape(1, FKP_opt.image_size, FKP_opt.image_size)
        image = torch.from_numpy(image)

        if keypoints is not None:
            keypoints = torch.from_numpy(keypoints)
            return {'image': image, 'keypoints': keypoints}
        else:
            return {'image': image}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5, data_type='ALL'):
        self.p = p
        self.data_type = data_type

    def __call__(self, sample):
        if self.data_type == 'large_kpset':
            flip_indices = [(0, 2), (1, 3)]
        elif self.data_type == 'small_kpset':
            flip_indices = [(0, 4), (1, 5), (2, 6), (3, 7),
                            (8, 12), (9, 13), (10, 14), (11, 15),
                            (16, 18), (17, 19)]
        else:
            flip_indices = [(0, 2), (1, 3),
                            (4, 8), (5, 9), (6, 10), (7, 11),
                            (12, 16), (13, 17), (14, 18), (15, 19),
                            (22, 24), (23, 25)]

        image, keypoints = sample['image'], sample['keypoints']

        if np.random.random() < self.p:
            image = image[:, ::-1]
            if keypoints is not None:
                for a, b in flip_indices:
                    keypoints[a], keypoints[b] = keypoints[b], keypoints[a]
                keypoints[::2] = 96. - keypoints[::2]

        return {'image': image, 'keypoints': keypoints}


def show_images(data, sample_num, idx_range, ncols=5, figsize=(15,10), with_keypoints=True, save=False):
    target = [id for id in range(idx_range)]
    indexs = random.sample(target, sample_num)
    plt.figure(figsize=figsize)
    nrows = len(indexs) // ncols + 1
    for i, idx in enumerate(indexs):
        image = np.fromstring(data.loc[idx, 'Image'], sep=' ').astype(np.float32).reshape(-1, FKP_opt.image_size)
        keypoints = data.loc[idx].drop('Image').values.astype(np.float32).reshape(-1, 2) if with_keypoints else []
        plt.subplot(nrows, ncols, i + 1)
        plt.title(f'Sample #{idx}')
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(image, cmap='gray')
        if len(keypoints):
            plt.scatter(keypoints[:, 0], keypoints[:, 1], s=24, marker='.', c='r')
    plt.show()


def generate_pred_kps(columns, test_set, predictions, image_ids=range(1,6)):
    pred = pd.DataFrame(predictions, columns=columns)
    pred = pd.concat([pred, test_set], axis=1)
    pred = pred.set_index('ImageId')
    return pred
