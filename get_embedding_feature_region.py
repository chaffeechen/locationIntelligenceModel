import os,sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn, cuda
import models.location_recommendation as rsmodels
from utils import load_model

from models.utils import *
from header import *
# from torch.utils.data import DataLoader

pjoin = os.path.join
#not used @this version /home/ubuntu/location_recommender_system/ /Users/yefeichen/Database/location_recommender_system

model_name = '' #same as main cmd --model XXX
wework_location_only = True



bid = 'atlas_location_uuid'
cid = 'duns_number'
sfx = ['','_right']

#=============================================================================================================================
#main
#=============================================================================================================================
def main():
    #cmd and arg parser
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--mode', choices=['input_grad'], default='input_grad')
    arg('--run_root', default='result/location_RSRBv4_191114/')
    arg('--path',default='/Users/yefeichen/Database/location_recommender_system/')
    arg('--model', default='location_recommend_region_model_v5')
    arg('--ckpt', type=str, default='model_loss_best.pt')
    arg('--maxK',type=int,default=50)
    arg('--apps',type=str,default='_191114.csv')
    arg('--pre_name', type=str, default='sampled_ww_')
    arg('--dbname',type=str,default='tmp_table')
    arg('--ww',action='store_true',help='produce ww location only')
    arg('--lscard', default='location_scorecard_200106.csv')

    #cuda version T/F
    use_cuda = cuda.is_available()

    args = parser.parse_args()
    #run_root: model/weights root
    run_root = Path(args.run_root)
    datapath = args.path
    datapath_mid = pjoin(datapath,args.dbname)

    # df_loc_feat = pd.read_csv(pjoin(TR_DATA_ROOT, 'location_feat' + args.apps), index_col=0)
    df_comp_feat = pd.read_csv(pjoin(datapath_mid, 'company_feat' + args.apps), index_col=0)
    # citynameabbr = ['PA', 'SF', 'SJ', 'LA', 'NY']
    # cityname = ['Palo Alto', 'San Francisco', 'San Jose', 'Los Angeles', 'New York']
    clfile = [c + args.apps for c in citynameabbr]

    not_cols = ['duns_number', 'atlas_location_uuid', 'label', 'city']

    if args.ww:
        loc_feat = pd.read_csv(pjoin(datapath, args.lscard))[['atlas_location_uuid', 'is_wework']]
        loc_ww = loc_feat.loc[loc_feat['is_wework'] == True]
        print('Total %d locations inside ls card belonged to ww.' % len(loc_ww))

    global model_name
    model_name = args.model

    model = getattr(rsmodels, args.model)(feat_comp_dim=102)
    md_path = Path(str(run_root) + '/' + args.ckpt)
    if md_path.exists():
        print('load weights from md_path')
        load_model(model, md_path)

    if use_cuda:
        model = model.cuda()

    model.eval()

    emb_vecs_all = []
    locName = []

    for ind_city, filename in enumerate(clfile):
        print('prcessing city %s'%filename)
        cldat = pd.read_csv(pjoin(datapath_mid, filename))
        if args.ww:
            cldat = cldat.merge(loc_ww,on=bid,suffixes=sfx)
        print('%d pairs in real' % len(cldat))
        # cldat['city'] = ind_city

        fn = lambda obj: obj.loc[np.random.choice(obj.index, args.maxK, True), :]
        tbB = cldat.groupby('atlas_location_uuid').apply(fn).reset_index(drop=True)[['duns_number', 'atlas_location_uuid']]

        list_col = list(df_comp_feat.columns)
        list_col = [col for col in list_col if col not in not_cols]
        tbB = tbB.merge(df_comp_feat,how='left',on='duns_number',suffixes=['','_right'])
        print('%d pairs to calc'%len(tbB))

        featRegion = tbB[list_col].to_numpy()
        tbBLoc = tbB[['atlas_location_uuid']]
        tbBLoc = tbBLoc.iloc[::args.maxK,:]
        tbBLoc['city'] = cityname[ind_city]

        featRegion = torch.FloatTensor(featRegion)

        if use_cuda:
            featRegion = featRegion.cuda()

        N, featdim = featRegion.shape

        featRegion = featRegion.view(-1, args.maxK, featdim)  # B,K,D

        outputs = model(feat_comp=None, feat_K_comp=featRegion, feat_loc=None)

        emb_vecs = outputs['feat_region_org']
        # print(emb_vecs.shape)

        loc_num,feat_dim = emb_vecs.shape
        print('loc num:%d,feature dims:%d'%(loc_num,feat_dim))

        emb_vecs_all.append(emb_vecs)
        locName.append(tbBLoc)
        assert(emb_vecs.shape[0]==len(tbBLoc))

    emb_vecs = torch.cat(emb_vecs_all,dim=0)
    locName = pd.concat(locName,axis=0).reset_index(drop=True)

    emb_vecs = emb_vecs.data.cpu().numpy()

    feat_cols = ['feat#' + str(c) for c in range(feat_dim)]
    feat_dat = pd.DataFrame(emb_vecs, columns=feat_cols)

    assert(len(locName)==len(feat_dat))


    loc_dat = pd.concat([locName, feat_dat], axis=1)
    loc_dat.to_csv(pjoin(datapath_mid,'location_feat_emb_'+args.model+'.csv'))














#=============================================================================================================================
#End of main
#=============================================================================================================================

# #=============================================================================================================================
# #predict
# #=============================================================================================================================
def _reduce_loss(loss):
    return loss.sum() / loss.shape[0]


def softmax_loss(results, labels):
    labels = labels.view(-1)
    loss = F.cross_entropy(results, labels, reduce=True)

    return loss


def softmax_lossV2(results,labels):

    softmax_label = labels[labels < N_CLASSES].view(-1)
    label_len = softmax_label.shape[0]
    softmax_results = results[:label_len,:]
    assert(label_len%2==0)
    loss = F.cross_entropy(softmax_results,softmax_label,reduce=True)

    return loss




if __name__ == '__main__':
    main()


