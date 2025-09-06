import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os


class EEG_Alcoholism:
    def __init__(self, standardize=True, test_size=0.2, random_seed=42, verbose=True):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_path = os.path.join(project_root, 'experiment', "data", "EEG_alcoholism", "SMNI_CMI_TRAIN")
        test_path = os.path.join(project_root, 'experiment', "data", "EEG_alcoholism", "SMNI_CMI_TEST")

        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

        if verbose:
            print('Reading train dataset...')
        for subdir in os.listdir(train_path):
            subdir_path = os.path.join(train_path, subdir)

            if not os.path.isdir(subdir_path):
                continue

            label = 1 if subdir[3] == 'a' else -1
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                df = pd.read_csv(file_path,
                                 compression='gzip',
                                 sep=r'\s+',
                                 comment='#',
                                 header=None,
                                 names=['trial', 'channel', 'sample', 'voltage'])
                self.X_train.append(df.pivot(index='sample', columns='channel', values='voltage'))
                self.y_train.append(label)

        if verbose:
            print('Reading test dataset...')
        for subdir in os.listdir(test_path):
            subdir_path = os.path.join(test_path, subdir)

            if not os.path.isdir(subdir_path):
                continue

            label = 1 if subdir[3] == 'a' else -1
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                df = pd.read_csv(file_path,
                                 compression='gzip',
                                 sep=r'\s+',
                                 comment='#',
                                 header=None,
                                 names=['trial', 'channel', 'sample', 'voltage'])
                self.X_test.append(df.pivot(index='sample', columns='channel', values='voltage'))
                self.y_test.append(label)

        self.X = np.vstack((self.X_train, self.X_test))
        self.y = np.hstack((self.y_train, self.y_test))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                random_state=random_seed, shuffle=True)

        if standardize:
            self.m = np.mean(np.vstack((self.X_train, self.X_test)), axis=0)
            self.s = np.std(np.vstack((self.X_train, self.X_test)), axis=0)
            self.X = (self.X - self.m) / self.s
            self.X_train = (self.X_train - self.m) / self.s
            self.X_test = (self.X_test - self.m) / self.s

    def get_train(self):
        return self.X_train, self.y_train

    def get_test(self):
        return self.X_test, self.y_test

    def get_all(self):
        return self.X, self.y

    def get_mean_sd(self):
        return self.m, self.s


class INRIA:
    def __init__(self, standardize=True, test_size=0.3, random_seed=42, noise=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        self.X = np.load(os.path.join(current_dir, 'INRIA Person Dataset', 'X_array.npy'))
        self.y = np.load(os.path.join(current_dir, 'INRIA Person Dataset', 'y_array.npy'))

        if noise == 'gaussian':
            n, p, q = self.X.shape
            np.random.seed(random_seed)
            noise = np.random.normal(0, 10, (n, p, q))
            self.X = self.X + noise
            self.X = np.clip(self.X, 0, 255)
        elif noise == 'spn':
            self.X = self.add_salt_pepper_noise(self.X, amount=0.10, salt_vs_pepper=0.5)

        if standardize:
            self.m = np.mean(self.X, axis=0)
            self.s = np.std(self.X, axis=0)
            self.X = (self.X - self.m) / self.s

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                shuffle=True, random_state=random_seed)

    def add_salt_pepper_noise(self, images, amount=0.10, salt_vs_pepper=0.5):
        """
        images: ndarray of shape (n, p, q), 值范围 0-255 或 0-1
        amount: 噪声比例（总共被替换的像素比例）
        salt_vs_pepper: 盐(白)的比例，比如0.5表示盐和椒各一半
        """
        noisy = images.copy()
        n, p, q = noisy.shape
        total_pixels = p * q

        num_salt = int(np.ceil(amount * total_pixels * salt_vs_pepper))
        num_pepper = int(np.ceil(amount * total_pixels * (1.0 - salt_vs_pepper)))

        for i in range(n):
            # 盐 (白色)
            np.random.seed(i)
            coords = (np.random.randint(0, p, num_salt), np.random.randint(0, q, num_salt))
            noisy[i][coords] = 255  # 255 或 1

            # 椒 (黑色)
            coords = (np.random.randint(0, p, num_pepper), np.random.randint(0, q, num_pepper))
            noisy[i][coords] = 1  # 0

        return noisy

    def get_all(self):
        return self.X, self.y

    def get_train(self):
        return self.X_train, self.y_train

    def get_test(self):
        return self.X_test, self.y_test


class Caltech:
    def __init__(self, standardize=True, test_size=0.3, random_seed=42, classes='dolphin_vs_butterfly', noise=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if classes == 'dolphin_vs_butterfly':
            self.X = np.load(os.path.join(current_dir, 'caltech-101', 'dolphin_vs_butterfly_x.npy'))
            self.y = np.load(os.path.join(current_dir, 'caltech-101', 'dolphin_vs_butterfly_y.npy'))
        elif classes == 'dolphin_vs_camera':
            self.X = np.load(os.path.join(current_dir, 'caltech-101', 'dolphin_vs_camera_x.npy'))
            self.y = np.load(os.path.join(current_dir, 'caltech-101', 'dolphin_vs_camera_y.npy'))
        elif classes == 'dolphin_vs_cup':
            self.X = np.load(os.path.join(current_dir, 'caltech-101', 'dolphin_vs_cup_x.npy'))
            self.y = np.load(os.path.join(current_dir, 'caltech-101', 'dolphin_vs_cup_y.npy'))
        elif classes == 'camera_vs_butterfly':
            self.X = np.load(os.path.join(current_dir, 'caltech-101', 'camera_vs_butterfly_x.npy'))
            self.y = np.load(os.path.join(current_dir, 'caltech-101', 'camera_vs_butterfly_y.npy'))
        elif classes == 'camera_vs_cup':
            self.X = np.load(os.path.join(current_dir, 'caltech-101', 'camera_vs_cup_x.npy'))
            self.y = np.load(os.path.join(current_dir, 'caltech-101', 'camera_vs_cup_y.npy'))
        elif classes == 'cup_vs_butterfly':
            self.X = np.load(os.path.join(current_dir, 'caltech-101', 'cup_vs_butterfly_x.npy'))
            self.y = np.load(os.path.join(current_dir, 'caltech-101', 'cup_vs_butterfly_y.npy'))

        if noise == 'gaussian':
            n, p, q = self.X.shape
            np.random.seed(random_seed)
            noise = np.random.normal(0, 10, (n, p, q))
            self.X = self.X + noise
            self.X = np.clip(self.X, 0, 255)
        elif noise == 'spn':
            self.X = self.add_salt_pepper_noise(self.X, amount=0.10, salt_vs_pepper=0.5)

        if standardize:
            self.m = np.mean(self.X, axis=0)
            self.s = np.std(self.X, axis=0)
            self.X = (self.X - self.m) / self.s

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                shuffle=True, random_state=random_seed)

    def get_all(self):
        return self.X, self.y

    def get_train(self):
        return self.X_train, self.y_train

    def get_test(self):
        return self.X_test, self.y_test

    def add_salt_pepper_noise(self, images, amount=0.10, salt_vs_pepper=0.5):
        """
        images: ndarray of shape (n, p, q), 值范围 0-255 或 0-1
        amount: 噪声比例（总共被替换的像素比例）
        salt_vs_pepper: 盐(白)的比例，比如0.5表示盐和椒各一半
        """
        noisy = images.copy()
        n, p, q = noisy.shape
        total_pixels = p * q

        num_salt = int(np.ceil(amount * total_pixels * salt_vs_pepper))
        num_pepper = int(np.ceil(amount * total_pixels * (1.0 - salt_vs_pepper)))

        for i in range(n):
            # 盐 (白色)
            np.random.seed(i)
            coords = (np.random.randint(0, p, num_salt), np.random.randint(0, q, num_salt))
            noisy[i][coords] = 255  # 255 或 1

            # 椒 (黑色)
            coords = (np.random.randint(0, p, num_pepper), np.random.randint(0, q, num_pepper))
            noisy[i][coords] = 1  # 0

        return noisy


class BCI_IV_IIa:
    def __init__(self, standardize=True, test_size=0.3, random_seed=42, classes='dolphin_vs_butterfly', subject='A01T'):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path_x = os.path.join(current_dir, 'BCI_IV_IIa', 'numpy_array', subject, classes+'_x.npy')
        path_y = os.path.join(current_dir, 'BCI_IV_IIa', 'numpy_array', subject, classes + '_y.npy')

        self.X = np.load(path_x)
        self.y = np.load(path_y)

        if standardize:
            self.m = np.mean(self.X, axis=0)
            self.s = np.std(self.X, axis=0)
            self.X = (self.X - self.m) / self.s

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                shuffle=True, random_state=random_seed)

    def get_all(self):
        return self.X, self.y

    def get_train(self):
        return self.X_train, self.y_train

    def get_test(self):
        return self.X_test, self.y_test