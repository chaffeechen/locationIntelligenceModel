from pathlib import Path
from typing import Callable, List
import random
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip)
import math
from udf.basic import timer

# image_size = 256

sfx = ['','_right']

# =======================================================================================================================
# data loader function for company location score
# =======================================================================================================================
class TrainDatasetLocationRS(Dataset):
    def __init__(self, df_comp_feat: pd.DataFrame,
                 df_loc_feat: pd.DataFrame,
                 df_pair: pd.DataFrame,
                 df_ensemble_score, flag_ensemble: bool,
                 emb_dict: dict, citynum=5,
                 name: str = 'train', posN=100, negN=200, testStep=500000):
        super().__init__()
        self._df_comp_feat = df_comp_feat.fillna(0)
        self._df_loc_feat = df_loc_feat.fillna(0)
        self._df_pair = df_pair.reset_index()
        self._df_ensemble_score = df_ensemble_score.reset_index(drop=True)
        self._name = name
        self._posN = posN
        self._negN = negN
        self._step = testStep
        self._emb_dict = emb_dict
        self._flag_ensemble = flag_ensemble
        self._not_cols = ['duns_number', 'atlas_location_uuid', 'label', 'city']

    def __len__(self):
        if self._name == 'train':
            return 1000
        else:
            return math.ceil(len(self._df_pair) / self._step)  # len of pair

    def tbatch(self):
        return self._posN + self._negN

    def __getitem__(self, idx: int):
        if self._name == 'train':
            # sample a part of data from training pair as positive seed
            dat1 = self._df_pair.sample(n=self._posN).reset_index(drop=True)
            dat2 = dat1.sample(frac=1).reset_index(drop=True)

            # generate negative sample from positive seed
            twin_dat = pd.merge(dat1, dat2, on='city', how='left', suffixes=['_left', '_right'])
            # twin_dat = twin_dat[twin_dat['atlas_location_uuid_left'] != twin_dat['atlas_location_uuid_right']]
            pot_neg_datA = twin_dat[
                ['duns_number_left', 'atlas_location_uuid_right']] \
                .rename(columns={'duns_number_left': 'duns_number', 'atlas_location_uuid_right': 'atlas_location_uuid'})

            pot_neg_datB = twin_dat[
                ['duns_number_right', 'atlas_location_uuid_left']] \
                .rename(columns={'duns_number_right': 'duns_number', 'atlas_location_uuid_left': 'atlas_location_uuid'})

            pot_neg_dat = pd.concat([pot_neg_datA, pot_neg_datB], axis=0)
            pot_neg_dat['label'] = 0
            dat1['label'] = 1

            # col alignment
            col_list = ['duns_number', 'atlas_location_uuid', 'label']
            dat1 = dat1[col_list]
            pot_neg_dat = pot_neg_dat[col_list]

            # clean pos dat in neg dat
            neg_dat = pd.merge(pot_neg_dat, dat1, on=['duns_number', 'atlas_location_uuid'], how='left',
                               suffixes=['', '_right']).reset_index(drop=True)
            neg_dat['label'] = neg_dat['label'].fillna(0)
            neg_dat = neg_dat[neg_dat['label_right'] != 1]

            neg_dat = neg_dat[['duns_number', 'atlas_location_uuid', 'label']].sample(
                n=min(self._negN, len(neg_dat))).reset_index(drop=True)

            pos_dat = dat1[col_list]
            res_dat = pd.concat([pos_dat, neg_dat], axis=0)
            res_dat = res_dat.sample(frac=1).reset_index(drop=True)
            Label = res_dat[['label']].to_numpy()
        else:
            inds = idx * self._step
            inde = min((idx + 1) * self._step, len(self._df_pair)) - 1  # loc[a,b] = [a,b] close set!!
            # res_dat = self._df_pair.loc[inds:inde,['duns_number','atlas_location_uuid','groundtruth']]
            # Label = (res_dat['atlas_location_uuid'] == res_dat['groundtruth']).to_numpy() + 0
            res_dat = self._df_pair.loc[inds:inde, ['duns_number', 'atlas_location_uuid', 'label']]
            Label = res_dat[['label']].to_numpy()

        # concate training pair with location/company feature
        F_res_dat = pd.merge(res_dat, self._df_comp_feat, on='duns_number', how='left')
        list_col = list(self._df_comp_feat.columns)
        list_col = [col for col in list_col if col not in self._not_cols]
        FeatComp = F_res_dat[list_col].to_numpy()

        F_res_dat = pd.merge(res_dat, self._df_loc_feat, on='atlas_location_uuid', how='left')
        list_col = list(self._df_loc_feat.columns)
        list_col = [col for col in list_col if col not in self._not_cols]
        # print(list_col)
        FeatLoc = F_res_dat[list_col].to_numpy()

        if self._flag_ensemble:
            F_res_dat = pd.merge(res_dat, self._df_ensemble_score, on=['atlas_location_uuid', 'duns_number'],
                                 how='left')
            list_col = list(self._df_ensemble_score.columns)
            list_col = [col for col in list_col if col not in self._not_cols]
            # print(list_col)
            FeatEnsembleScore = F_res_dat[list_col].to_numpy()
        else:
            FeatEnsembleScore = np.ones((len(F_res_dat), 1), dtype=np.float32)

        # trans id(str) 2 Long
        loc_name_str = res_dat['atlas_location_uuid'].values.tolist()
        loc_name_int = [self._emb_dict[n] for n in loc_name_str]

        # [B,Len_feat],[B,1]
        assert (len(Label) == len(FeatComp) and len(Label) == len(FeatLoc))
        # print(Label.sum(), FeatLoc.sum(),FeatComp.sum())

        featComp = torch.FloatTensor(FeatComp)
        featLoc = torch.FloatTensor(FeatLoc)
        featEnsembleScore = torch.FloatTensor(FeatEnsembleScore)
        featId = torch.LongTensor(loc_name_int).reshape(-1, 1)
        target = torch.LongTensor(Label).reshape(-1, 1)

        return {"feat_comp": featComp,
                "feat_loc": featLoc,
                "target": target,
                "feat_id": featId,
                "feat_ensemble_score": featEnsembleScore,
                "feat_comp_dim": FeatComp.shape,
                "feat_loc_dim": FeatLoc.shape}


def collate_TrainDatasetLocationRS(batch):
    """
    special collate_fn function for UDF class TrainDatasetTriplet
    :param batch: 
    :return: 
    """
    feat_comp = []
    feat_loc = []
    feat_id = []
    feat_ensemble_score = []
    labels = []

    for b in batch:
        feat_comp.append(b['feat_comp'])
        feat_loc.append(b['feat_loc'])
        feat_id.append(b['feat_id'])
        feat_ensemble_score.append(b['feat_ensemble_score'])
        labels.append(b['target'])

    feat_comp = torch.cat(feat_comp, 0)
    feat_loc = torch.cat(feat_loc, 0)
    feat_id = torch.cat(feat_id, 0)
    feat_ensemble_score = torch.cat(feat_ensemble_score, 0)
    labels = torch.cat(labels, 0)
    # print(feat_comp.shape,feat_loc.shape,labels.shape)

    assert (feat_loc.shape[0] == labels.shape[0])
    assert (feat_comp.shape[0] == labels.shape[0])
    assert (feat_id.shape[0] == labels.shape[0])
    assert (feat_ensemble_score.shape[0] == labels.shape[0])
    return {
        "feat_comp": feat_comp,
        "feat_loc": feat_loc,
        "feat_id": feat_id,
        "feat_ensemble_score": feat_ensemble_score,
        "target": labels
    }


# =======================================================================================================================
# data loader function for company location region modelling
# RSRB: Recommendation System Region Based
# =======================================================================================================================
class TrainDatasetLocationRSRB(Dataset):
    def __init__(self, df_comp_feat: pd.DataFrame,
                 df_loc_feat: pd.DataFrame,
                 df_pair: pd.DataFrame,
                 citynum=5,
                 name: str = 'train', trainStep=10000, testStep=500000):
        super().__init__()
        self._df_comp_feat = df_comp_feat.fillna(0)
        self._df_loc_feat = df_loc_feat.fillna(0)
        self._df_pair = df_pair.reset_index()
        self._name = name
        self._step = testStep
        self._citynum = citynum
        self._maxK = 100
        self._traintimes = trainStep
        self.cldat = []
        self.locname = []
        self.df_comp_feat_city = []
        if name in ['train', 'train_fast']:
            for ind_city in range(citynum):
                self.cldat.append(self._df_pair[(self._df_pair['fold'] == 0) & (self._df_pair['city'] == ind_city)])
                self.locname.append(self.cldat[ind_city].groupby('atlas_location_uuid').head(1).reset_index(drop=True)[
                                        ['atlas_location_uuid']])
                self.df_comp_feat_city.append(
                    self._df_comp_feat[self._df_comp_feat['city'] == ind_city].reset_index(drop=True))
        self._debug = False
        self._not_cols = ['duns_number', 'atlas_location_uuid', 'label', 'city']

    def __len__(self):
        if self._name in ['train', 'train_fast']:
            return self._traintimes
        else:
            return math.ceil(len(self._df_pair) / self._step)  # len of pair

    def tbatch(self):
        return 0

    def __getitem__(self, idx: int):
        tc = timer(display=self._debug)
        if self._name == 'train':
            pass
            # #pick a city randomly
            # ind_city = math.floor(random.random() * self._citynum)
            # cldat = self.cldat[ind_city]
            #
            # fn = lambda obj: obj.loc[np.random.choice(obj.index, 1, True), :]
            # cldatGrp = cldat.groupby('atlas_location_uuid')
            # tbA = cldatGrp.apply(fn).reset_index(drop=True)[
            #     ['duns_number', 'atlas_location_uuid']]
            # # print('1.len of tbA %d:' % len(tbA))
            # fn = lambda obj: obj.loc[np.random.choice(obj.index, self._maxK, True), :]
            # tbB = cldatGrp.apply(fn).reset_index(drop=True)[
            #     ['duns_number', 'atlas_location_uuid']]
            # # print('1.len of tbB %d' % len(tbB))
            #
            # ###======================Pos=============================###
            # tbA['mk'] = 'A'
            # tbB = tbB.merge(tbA, on=['duns_number', 'atlas_location_uuid'], how='left', suffixes=['', '_right'])
            # tbB = tbB[tbB['mk'].isnull()]
            # # print('2.len of tbB not included in tbA %d' % len(tbB))
            # # we need to full fill the data
            # tbBGrp = tbB.groupby('atlas_location_uuid')
            # tbB = tbBGrp.apply(fn).reset_index(drop=True)[
            #     ['duns_number', 'atlas_location_uuid']]
            # tbB['mk'] = 'B'
            # # print('3.len of tbB full filled again %d' % len(tbB))
            # # in case tbB cut some locations from tbA, lets shrink tbA
            # tblocB = tbBGrp.first().reset_index()
            # tblocB['mk'] = 'B'
            # # print('4.len of locations in tbB %d' % len(tblocB))
            # tbA = tbA.merge(tblocB, on='atlas_location_uuid', how='left', suffixes=['', '_right'])
            # tbA = tbA[tbA['mk_right'].notnull()][['duns_number', 'atlas_location_uuid', 'mk']].reset_index(drop=True)
            # # print('4.len of tbA with common locations of tbB %d' % len(tbA))
            #
            # ###======================Neg=============================###
            # tbAA = pd.concat([tbA, tbA.sample(frac=1).reset_index() \
            #                  .rename(
            #     columns={'duns_number': 'duns_number_n', 'atlas_location_uuid': 'atlas_location_uuid_n', 'mk': 'mk_n'})]
            #                  , axis=1)
            # # print('5.len of negpair %d' % len(tbAA))
            # tbAA = tbAA.merge(cldat, \
            #                   left_on=['duns_number_n', 'atlas_location_uuid'],
            #                   right_on=['duns_number', 'atlas_location_uuid'], \
            #                   how='left', suffixes=['', '_right'])
            #
            # tbC = tbAA[tbAA['duns_number_right'].isnull()][['duns_number_n', 'atlas_location_uuid']] \
            #     .rename(columns={'duns_number_n': 'duns_number'})
            # # print('6.len of neg data %d' % len(tbC))
            #
            # # in case tbC cut some locations from tbA and tbB
            # tbC['mk'] = 'C'
            # tblocC = tbC.groupby('atlas_location_uuid').first().reset_index()
            # # print('6.locations in neg data %d' % len(tblocC))
            # tbA = tbA.merge(tblocC, on='atlas_location_uuid', how='left', suffixes=['', '_right'])
            # tbA = tbA[tbA['mk_right'].notnull()][['duns_number', 'atlas_location_uuid', 'mk']].reset_index(drop=True)
            # # print('final tbA len %d' % len(tbA))
            #
            # tbB = tbB.merge(tblocC, on='atlas_location_uuid', how='left', suffixes=['', '_right'])
            # tbB = tbB[tbB['mk_right'].notnull()][['duns_number', 'atlas_location_uuid', 'mk']].reset_index(drop=True)
            # # print('final tbB len %d' % len(tbB))
            #
            # tbA = tbA.sort_values(by='atlas_location_uuid')
            # tbB = tbB.sort_values(by='atlas_location_uuid')
            # tbC = tbC.sort_values(by='atlas_location_uuid')
            #
            # assert (len(tbA) == len(tbC) and len(tbB) == len(tbA) * self._maxK)
            #
            # list_col = list(self._df_comp_feat.columns)
            # list_col = [col for col in list_col if col not in ['duns_number', 'atlas_location_uuid', 'label','city']]
            #
            # featA = tbA.merge(self._df_comp_feat,on='duns_number',how='left',suffixes=['','_right'])[list_col]
            # featB = tbB.merge(self._df_comp_feat,on='duns_number',how='left',suffixes=['','_right'])[list_col]
            # featC = tbC.merge(self._df_comp_feat,on='duns_number',how='left',suffixes=['','_right'])[list_col]
        elif self._name == 'train_fast':
            num_building_batch = 20 # select num_building_batch buildings for each batch
            num_pos = 50  # for each building in the batch select num_pos companies as positive samples
            num_region = self._maxK

            data_batch = num_pos + num_pos * num_region
            num_pos_pair = num_pos * num_building_batch
            # num_neg_pair = 2*num_pos_pair

            ind_city = math.floor(random.random() * self._citynum)
            cldat = self.cldat[ind_city]

            tc.start(it='location selection')
            smp_loc_name = self.locname[ind_city].sample(n=num_building_batch).reset_index(drop=True)
            smp_cldat = cldat.merge(smp_loc_name, on='atlas_location_uuid', how='inner', suffixes=['', '_right'])
            tc.eclapse()

            tc.start(it='sample pos and region data')
            cldatGrp = smp_cldat.groupby('atlas_location_uuid')
            tbAB = cldatGrp.apply(lambda x: x.sample(data_batch, replace=True)).reset_index(drop=True)[
                ['duns_number', 'atlas_location_uuid']]
            tc.eclapse()

            tc.start(it='create tbA:pos company and tbB:companies inside region')
            tbABGrp = tbAB.groupby('atlas_location_uuid')
            tbA = tbABGrp.head(num_pos).reset_index(drop=True)

            tbB = tbABGrp.tail(num_pos * num_region).reset_index(drop=True)
            tc.eclapse()

            assert (len(tbA) == num_pos_pair)

            tc.start(it='get location neg pairs')
            smp_loc_name_pair1 = \
                pd.concat([smp_loc_name,
                           smp_loc_name.sample(frac=1, replace=False).reset_index(drop=True) \
                          .rename(columns={'atlas_location_uuid': 'atlas_location_uuid_neg'})], axis=1)

            smp_loc_name_pair2 = \
                pd.concat([smp_loc_name,
                           smp_loc_name.sample(frac=1, replace=False).reset_index(drop=True) \
                          .rename(columns={'atlas_location_uuid': 'atlas_location_uuid_neg'})], axis=1)

            smp_loc_name_pair = pd.concat([smp_loc_name_pair1,smp_loc_name_pair2],axis=0)

            tbC = \
            smp_loc_name_pair.merge(tbA, left_on='atlas_location_uuid_neg', right_on='atlas_location_uuid', how='inner',
                                    suffixes=['', '_useless'])[
                ['duns_number', 'atlas_location_uuid']].reset_index(drop=True)
            tc.eclapse()

            tc.start('sort')
            tbA = tbA.sort_values(by='atlas_location_uuid').reset_index(drop=True)
            tbB = tbB.sort_values(by='atlas_location_uuid').reset_index(drop=True)
            tbC = tbC.sort_values(by='atlas_location_uuid').reset_index(drop=True)
            tc.eclapse()

            list_col = list(self._df_comp_feat.columns)
            list_col = [col for col in list_col if col not in self._not_cols]

            tc.start('merge')
            tbACB = pd.concat([tbA, tbC, tbB], axis=0, sort=False).reset_index(drop=True)
            featACB_comp = \
            tbACB.merge(self.df_comp_feat_city[ind_city], on='duns_number', how='left', suffixes=sfx)[
                list_col]

            featA = featACB_comp.loc[:num_pos_pair - 1]
            featC = featACB_comp.loc[num_pos_pair:3 * num_pos_pair - 1]
            featB = featACB_comp.loc[3 * num_pos_pair:]

            assert (len(featC) == 2*len(featA))

            list_col = list(self._df_loc_feat.columns)
            list_col = [col for col in list_col if col not in self._not_cols]
            # tbA and tbB share the same location, thus tbA is used.
            featB_loc = tbA.merge(self._df_loc_feat, on='atlas_location_uuid', how='left', suffixes=sfx)[
                list_col]
            tc.eclapse()

        else:
            dataLen = len(self._df_pair[self._df_pair['mk'] == 'A'])
            dataLenB = len(self._df_pair[self._df_pair['mk'] == 'B'])
            inds = idx * self._step
            inde = min((idx + 1) * self._step, dataLen) - 1  # loc[a,b] = [a,b] close set!!
            # res_dat = self._df_pair.loc[inds:inde, ['duns_number', 'atlas_location_uuid','city','mk']]

            indsB = idx * self._step * self._maxK
            indeB = min((idx + 1) * self._step * self._maxK, dataLenB) - 1  # loc[a,b] = [a,b] close set!!

            list_col = list(self._df_comp_feat.columns)
            list_col = [col for col in list_col if col not in self._not_cols]

            datA = self._df_pair[self._df_pair['mk'] == 'A'].sort_values(
                by=['city', 'atlas_location_uuid']).reset_index(drop=True)
            datB = self._df_pair[self._df_pair['mk'] == 'B'].sort_values(
                by=['city', 'atlas_location_uuid']).reset_index(drop=True)
            datC = self._df_pair[self._df_pair['mk'] == 'C'].sort_values(
                by=['city', 'atlas_location_uuid']).reset_index(drop=True)

            datA = datA.loc[inds:inde, ['duns_number', 'atlas_location_uuid', 'city', 'mk']]
            datB = datB.loc[indsB:indeB, ['duns_number', 'atlas_location_uuid', 'city', 'mk']]
            datC = datC.loc[inds:inde, ['duns_number', 'atlas_location_uuid', 'city', 'mk']]

            featA = datA.merge(self._df_comp_feat, on='duns_number', how='left', suffixes=['', '_right'])[list_col]
            featB = datB.merge(self._df_comp_feat, on='duns_number', how='left', suffixes=['', '_right'])[list_col]
            featC = datC.merge(self._df_comp_feat, on='duns_number', how='left', suffixes=['', '_right'])[list_col]

            list_col = list(self._df_loc_feat.columns)
            list_col = [col for col in list_col if col not in self._not_cols]
            featB_loc = datA.merge(self._df_loc_feat, on='atlas_location_uuid', how='left', suffixes=['', '_right'])[
                list_col]

        # all branch need such operation...
        tc.start('Transfer storage')
        featA, featB, featC, featB_loc = featA.to_numpy(), featB.to_numpy(), featC.to_numpy(), featB_loc.to_numpy()

        featCompPos = torch.FloatTensor(featA)  # B,D
        featRegion = torch.FloatTensor(featB)
        featLoc = torch.FloatTensor(featB_loc)
        N, featdim = featRegion.shape
        # print(featA.shape,featB.shape,featC.shape)
        assert (N == featCompPos.shape[0] * self._maxK)

        featRegion = featRegion.view(-1, self._maxK, featdim)  # B,K,D

        featCompNeg = torch.FloatTensor(featC)  # B,D
        tc.eclapse()

            # featLoc = (torch.randn_like(featLoc) * disturbance + 1) * featLoc

        return {
            "feat_comp_pos": featCompPos,
            "feat_comp_neg": featCompNeg,
            "feat_comp_region": featRegion,
            "feat_loc": featLoc,
        }


def collate_TrainDatasetLocationRSRB(batch):
    """
    special collate_fn function for UDF class TrainDatasetTriplet
    :param batch: 
    :return: 
    """
    feat_comp_pos = []
    feat_comp_neg = []
    feat_comp_region = []
    feat_loc = []

    for b in batch:
        feat_comp_pos.append(b['feat_comp_pos'])
        feat_comp_neg.append(b['feat_comp_neg'])
        feat_comp_region.append(b['feat_comp_region'])
        feat_loc.append(b['feat_loc'])

    feat_comp_pos = torch.cat(feat_comp_pos, 0)
    feat_comp_neg = torch.cat(feat_comp_neg, 0)
    feat_comp_region = torch.cat(feat_comp_region, 0)
    feat_loc = torch.cat(feat_loc, 0)
    # print(feat_comp.shape,feat_loc.shape,labels.shape)
    # 2 * feat_comp_pos.shape[0] == feat_comp_neg.shape[0] train true test wrong
    assert (feat_comp_region.shape[0] == feat_comp_pos.shape[0] and
            feat_loc.shape[0] == feat_comp_pos.shape[0])

    return {
        "feat_comp_pos": feat_comp_pos,
        "feat_comp_neg": feat_comp_neg,
        "feat_comp_region": feat_comp_region,
        "feat_loc": feat_loc,
    }

class TestDatasetLocationRSRB(Dataset):
    def __init__(self, df_comp_feat: pd.DataFrame,
                 df_loc_feat: pd.DataFrame,
                 df_region_feat:pd.DataFrame,
                 df_pair: pd.DataFrame, testStep=500000):
        super().__init__()
        self._df_comp_feat = df_comp_feat.fillna(0)
        self._df_loc_feat = df_loc_feat.fillna(0)
        self._df_region_feat = df_region_feat.fillna(0)
        self._df_pair = df_pair.reset_index()
        self._step = testStep
        self.cldat = []
        self.locname = []

        self._debug = False
        self._not_cols = ['duns_number', 'atlas_location_uuid', 'label', 'city']

    def __len__(self):
        return math.ceil(len(self._df_pair) / self._step)  # len of pair

    def tbatch(self):
        return 0

    def __getitem__(self, idx: int):
        tc = timer(display=self._debug)
        dataLen = len(self._df_pair)
        inds = idx * self._step
        inde = min((idx + 1) * self._step, dataLen) - 1

        datA = self._df_pair.loc[inds:inde, ['duns_number', 'atlas_location_uuid','label']]

        tc.start('Append feature with pairs')
        list_col = list(self._df_comp_feat.columns)
        list_col = [col for col in list_col if col not in self._not_cols]
        featA_comp = datA.merge(self._df_comp_feat, on='duns_number', how='left', suffixes=sfx)[list_col]

        list_col = list(self._df_region_feat.columns)
        list_col = [col for col in list_col if col not in self._not_cols]
        featA_region = datA.merge(self._df_region_feat, on='atlas_location_uuid', how='left', suffixes=sfx)[list_col]


        list_col = list(self._df_loc_feat.columns)
        list_col = [col for col in list_col if col not in self._not_cols]
        featA_loc = datA.merge(self._df_loc_feat, on='atlas_location_uuid', how='left', suffixes=sfx)[
            list_col]
        tc.eclapse()

        # all branch need such operation...
        tc.start('Transfer storage')
        featA_comp, featA_region, featA_loc = featA_comp.to_numpy(), featA_region.to_numpy(), featA_loc.to_numpy()

        featComp = torch.FloatTensor(featA_comp)  # B,D
        featRegion = torch.FloatTensor(featA_region)
        featLoc = torch.FloatTensor(featA_loc)
        targets = torch.LongTensor(datA[['label']].to_numpy().reshape(-1,1))

        N, featdim = featRegion.shape

        assert ( (N == featComp.shape[0]) and (N==featLoc.shape[0]) )
        tc.eclapse()

        return {
            "feat_comp": featComp,
            "feat_region": featRegion,
            "feat_loc": featLoc,
            "targets": targets,
        }


def collate_TestDatasetLocationRSRB(batch):
    """
    special collate_fn function for UDF class TrainDatasetTriplet
    :param batch: 
    :return: 
    """
    feat_comp = []
    feat_region = []
    feat_loc = []
    targets = []

    for b in batch:
        feat_comp.append(b['feat_comp'])
        feat_region.append(b['feat_region'])
        feat_loc.append(b['feat_loc'])
        targets.append(b['targets'])

    feat_comp = torch.cat(feat_comp, 0)
    feat_region = torch.cat(feat_region, 0)
    feat_loc = torch.cat(feat_loc, 0)
    targets = torch.cat(targets,0)
    # print(feat_comp.shape,feat_loc.shape,labels.shape)

    assert (feat_comp.shape[0] == feat_region.shape[0]  and
            feat_loc.shape[0] == feat_comp.shape[0])

    return {
        "feat_comp": feat_comp,
        "feat_region": feat_region,
        "feat_loc": feat_loc,
        "targets": targets,
    }


# =======================================================================================================================
# data loader function for company location score more general and id embedding
# =======================================================================================================================
class TrainDatasetLocationRS_salesforce(Dataset):
    def __init__(self, df_comp_feat: pd.DataFrame,
                 df_loc_feat: pd.DataFrame,
                 df_pair: pd.DataFrame,
                 df_city_atlas:pd.DataFrame,
                 df_acc_city:pd.DataFrame,
                 emb_dict: dict,
                 colname={'location':'atlas_location_uuid','company':'duns_number'},
                 name: str = 'train', posN=100, negN=200, testStep=500000):
        super().__init__()
        self._df_comp_feat = df_comp_feat.fillna(0)
        self._df_loc_feat = df_loc_feat.fillna(0)
        self._df_pair = df_pair.reset_index()
        self._name = name
        self._posN = posN
        self._negN = negN
        self._step = testStep
        self._emb_dict = emb_dict
        self._not_cols = ['label', 'city'] + [ v for k,v in colname.items()]
        self.cid = colname['company']
        self.bid = colname['location']
        self._df_city_atlas = df_city_atlas
        self._df_acc_city = df_acc_city

    def __len__(self):
        if self._name == 'train':
            return 1000
        else:
            return math.ceil(len(self._df_pair) / self._step)  # len of pair

    def tbatch(self):
        return self._posN + self._negN

    def __getitem__(self, idx: int):
        cid = str(self.cid)
        bid = str(self.bid)
        if self._name == 'train':
            # sample a part of data from training pair as positive seed
            dat1 = self._df_pair.sample(n=self._posN).reset_index(drop=True)
            pot_neg_dat = dat1.merge(self._df_acc_city, on = cid, suffixes= sfx).merge(self._df_city_atlas, on='city',suffixes=sfx)[[cid,bid]]
            pot_neg_dat['label'] = 0
            dat1['label'] = 1

            # col alignment
            col_list = [cid, bid, 'label']
            dat1 = dat1[col_list]
            pot_neg_dat = pot_neg_dat[col_list]

            # clean pos dat in neg dat
            neg_dat = pd.merge(pot_neg_dat, dat1, on=[cid, bid], how='left',
                               suffixes=sfx).reset_index(drop=True)
            neg_dat['label'] = neg_dat['label'].fillna(0)
            neg_dat = neg_dat.loc[neg_dat['label_right'] != 1,:]

            neg_dat = neg_dat[[cid, bid, 'label']].sample(
                n=min(self._negN, len(neg_dat))).reset_index(drop=True)

            pos_dat = dat1[col_list]
            res_dat = pd.concat([pos_dat, neg_dat], axis=0)
            res_dat = res_dat.sample(frac=1).reset_index(drop=True)
            Label = res_dat[['label']].to_numpy()
        else:
            inds = idx * self._step
            inde = min((idx + 1) * self._step, len(self._df_pair)) - 1  # loc[a,b] = [a,b] close set!!
            res_dat = self._df_pair[[cid,bid,'label']]
            res_dat = res_dat.iloc[inds:inde, :]
            Label = res_dat[['label']].to_numpy()

        # concate training pair with location/company feature
        F_res_dat = pd.merge(res_dat, self._df_comp_feat, on=cid, how='left',suffixes=sfx)
        list_col = list(self._df_comp_feat.columns)
        list_col = [col for col in list_col if col not in self._not_cols]
        FeatComp = F_res_dat[list_col].to_numpy()

        F_res_dat = pd.merge(res_dat, self._df_loc_feat, on=bid, how='left', suffixes=sfx)
        list_col = list(self._df_loc_feat.columns)
        list_col = [col for col in list_col if col not in self._not_cols]
        FeatLoc = F_res_dat[list_col].to_numpy()


        # trans id(str) 2 Long
        loc_name_str = res_dat[bid].values.tolist()
        loc_name_int = [self._emb_dict[n] for n in loc_name_str]

        # [B,Len_feat],[B,1]
        assert (len(Label) == len(FeatComp) and len(Label) == len(FeatLoc))
        # print(Label.sum(), FeatLoc.sum(),FeatComp.sum())

        featComp = torch.FloatTensor(FeatComp)
        featLoc = torch.FloatTensor(FeatLoc)
        featId = torch.LongTensor(loc_name_int).reshape(-1, 1)
        target = torch.LongTensor(Label).reshape(-1, 1)

        return {"feat_comp": featComp,
                "feat_loc": featLoc,
                "target": target,
                "feat_id": featId,
                "feat_comp_dim": FeatComp.shape,
                "feat_loc_dim": FeatLoc.shape}


def collate_TrainDatasetLocationRS_salesforce(batch):
    """
    special collate_fn function for UDF class TrainDatasetTriplet
    :param batch: 
    :return: 
    """
    feat_comp = []
    feat_loc = []
    feat_id = []
    feat_ensemble_score = []
    labels = []

    for b in batch:
        feat_comp.append(b['feat_comp'])
        feat_loc.append(b['feat_loc'])
        feat_id.append(b['feat_id'])
        labels.append(b['target'])

    feat_comp = torch.cat(feat_comp, 0)
    feat_loc = torch.cat(feat_loc, 0)
    feat_id = torch.cat(feat_id, 0)
    labels = torch.cat(labels, 0)
    # print(feat_comp.shape,feat_loc.shape,labels.shape)

    assert (feat_loc.shape[0] == labels.shape[0])
    assert (feat_comp.shape[0] == labels.shape[0])
    assert (feat_id.shape[0] == labels.shape[0])
    return {
        "feat_comp": feat_comp,
        "feat_loc": feat_loc,
        "feat_id": feat_id,
        "target": labels
    }

# =======================================================================================================================
# image name looks like : idx_copy.jpg
def get_ids(root: Path) -> List[str]:
    return sorted({p.name.split('_')[0] for p in root.glob('*.jpg')})

