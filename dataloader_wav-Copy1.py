import random
import numpy as np
import torch
import soundfile as sf
from torch.utils.data import Dataset
import python_speech_features


class WavDataset(Dataset):
    def __init__(self, utt2data, utt2label=None, label2int=None, need_aug=False, with_label=True, shuffle=True, feat='ComParE'):
        self.utt2data = utt2data
        self.dataset_size = len(self.utt2data)
        self.shuffle = shuffle
        self.with_label = with_label
        self.utt2label = utt2label
        self.label2int = label2int
        self.need_aug = need_aug
        self.feat = feat

        if self.with_label:
            assert self.utt2label and self.label2int is not None, "utt2label must be provided in with_label model! "

        if shuffle:
            random.shuffle(self.utt2data)
            
        sub_init(self.feat)

    def sub_init(self, feat):
        task_name = 'ComParE2019_OrcaActivity'
        feat_conf = {'ComParE':      (6373, 1, ';', 'infer'),
                 'BoAW-125':     ( 250, 1, ';',  None),
                 'BoAW-250':     ( 500, 1, ';',  None),
                 'BoAW-500':     (1000, 1, ';',  None),
                 'BoAW-1000':    (2000, 1, ';',  None),
                 'BoAW-2000':    (4000, 1, ';',  None),
                 'auDeep-40':    (1024, 2, ',', 'infer'),
                 'auDeep-50':    (1024, 2, ',', 'infer'),
                 'auDeep-60':    (1024, 2, ',', 'infer'),
                 'auDeep-70':    (1024, 2, ',', 'infer'),
                 'auDeep-fused': (4096, 2, ',', 'infer')}

        num_feat = feat_conf[feat][0]
        ind_off  = feat_conf[feat][1]
        sep      = feat_conf[feat][2]
        header   = feat_conf[feat][3]

        # Path of the features
        features_path = './features/'

        # Start
        print('\nRunning ' + task_name + ' ' + feat + ' baseline ... (this might take a while) \n')

        # Load features and labels
        if 'train' in self.utt2data[0][1]:
            X_total = pd.read_csv(features_path + task_name + '.' + feat + '.train.csv', sep=sep,
                                  header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
        elif 'test' in self.utt2data[0][1]:
            X_total = pd.read_csv(features_path + task_name + '.' + feat + '.test.csv', sep=sep,
                                  header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
        elif 'dev' in self.utt2data[0][1]:
            X_total = pd.read_csv(features_path + task_name + '.' + feat + '.devel.csv', sep=sep,
                                  header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
        self.X_return = []
        for i in range(len(X_total)):
            self.X_return.append(X_total[i][:6336].reshape(99,64))
            
    def __len__(self):
        return self.dataset_size

    def _transform_data(self, feat):
        return self.X_return


    def augment(self, o_sig, sr, utt_label):
        return o_sig

    def __getitem__(self, sample_idx):
        idx = int(sample_idx)
        utt, filename = self.utt2data[idx]

        feat = self._transform_data(self.feat)[idx]
        feat = torch.from_numpy(np.array(feat))

        if self.with_label:
            return utt, feat, int(self.label2int[self.utt2label[utt]])
        else:
            return utt, feat