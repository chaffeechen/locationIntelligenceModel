import os, sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
from itertools import islice
import json
from pathlib import Path
import shutil
import warnings
from typing import Dict
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
import torch
from torch import nn, cuda
from torch.optim import Adam, SGD
import tqdm
import models.location_recommendation as rsmodels
from dataset import TrainDatasetLocationRS_salesforce, collate_TrainDatasetLocationRS_salesforce
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from utils import (write_event, load_model, ThreadingDataLoader as DataLoader, adjust_learning_rate)

from models.utils import *
from udf.basic import save_obj, load_obj, calc_topk_acc_cat_all, topk_recall_score_all
import matplotlib.pyplot as plt
from gunlib.company_location_score_lib import reason_json_format,translocname2dict_general,translocname2dict

pjoin = os.path.join
# not used @this version
TR_DATA_ROOT = '/home/ubuntu/location_recommender_system/'
TT_DATA_ROOT = '/home/ubuntu/location_recommender_system/'


nPosTr = 1000
nNegTr = 2000

model_name = ''  # same as main cmd --model XXX
wework_location_only = True

colname = {
    'location':'atlas_location_uuid',
    'company':'duns_number',
}

comp_feat_file_prename = 'salesforce_company_feat'
loc_feat_file_prename = 'salesforce_location_feat'
train_valid_file_prename = 'train_val_test_location_company_82split'
salesforce_acc_city_prename = 'salesforce_acc_city'
salesforce_city_atlas_prename = 'salesforce_city_atlas'



# =============================================================================================================================
# main
# =============================================================================================================================
def main():
    # cmd and arg parser
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--mode', choices=['train', 'validate', 'predict_valid', 'predict_test',
                           'predict_salesforce'], default='train')
    arg('--run_root', default='result/salesforce_location_company') # where the model will be saved
    arg('--model', default='location_recommend_model_v3') # which version of model will be used
    arg('--ckpt', type=str, default='model_loss_best.pt') # name of checkpoint which should be used in validation/prediction
    arg('--batch-size', type=int, default=1)
    arg('--step', type=str, default=8)  # update the gradients every 8 batch(sample num = step*batch-size*inner_size)
    arg('--workers', type=int, default=16)
    arg('--lr', type=float, default=3e-4)
    arg('--patience', type=int, default=4)
    arg('--clean', action='store_true')
    arg('--n-epochs', type=int, default=80)
    arg('--epoch-size', type=int)
    arg('--cos_sim_loss', action='store_true')
    arg('--sample_rate', type=float, default=1.0)  # sample part of testing data for evaluating during training
    arg('--testStep', type=int, default=500000)
    arg('--query_location', action='store_true', help='use location as query')
    arg('--apps', type=str, default='_191114.csv')
    arg('--all', action='store_true', help='return all the prediction')
    arg('--data_root',default='/home/ubuntu/location_recommender_system/')
    arg('--ls_card',default='location_scorecard_200106.csv')

    # cuda version T/F
    use_cuda = cuda.is_available()

    args = parser.parse_args()
    # run_root: model/weights root
    run_root = Path(args.run_root)
    global TR_DATA_ROOT,TT_DATA_ROOT
    TR_DATA_ROOT = args.data_root
    TT_DATA_ROOT = args.data_root

    datapath = args.data_root
    salesforce_path = pjoin(datapath,'salesforce')

    global model_name
    model_name = args.model

    df_all_pair = pd.read_csv(pjoin(salesforce_path, train_valid_file_prename + args.apps), index_col=0)
    df_comp_feat = pd.read_csv(pjoin(salesforce_path, comp_feat_file_prename + args.apps), index_col=0)
    df_loc_feat = pd.read_csv(pjoin(salesforce_path, loc_feat_file_prename + args.apps), index_col=0)

    df_acc_city = pd.read_csv(pjoin(salesforce_path, salesforce_acc_city_prename + args.apps), index_col=0)
    df_city_atlas = pd.read_csv(pjoin(salesforce_path, salesforce_city_atlas_prename + args.apps), index_col=0)

    # split train/valid fold
    df_train_pair = df_all_pair[df_all_pair['fold'] == 0]

    if args.mode == 'train':
        df_valid_pair = df_all_pair[df_all_pair['fold'] == 2].sample(frac=args.sample_rate).reset_index(drop=True)
        print('Validate size%d' % len(df_valid_pair))
    else:
        df_valid_pair = df_all_pair[df_all_pair['fold'] == 2]
        print('Validate size%d' % len(df_valid_pair))
        if args.query_location:
            df_valid_pair = df_valid_pair.sort_values(by=[colname['location']]).reset_index(drop=True)
    del df_all_pair

    loc_name_dict = translocname2dict_general(df_loc_feat,colname=colname['location'])
    print('Location Embedding Number: %d' % len(loc_name_dict))

    ##::DataLoader
    def make_loader(df_pair: pd.DataFrame, name='train', testStep=args.testStep, shuffle=True) -> DataLoader:
        return DataLoader(
            TrainDatasetLocationRS_salesforce(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, df_pair=df_pair,
                                              df_city_atlas= df_acc_city,df_acc_city=df_city_atlas,
                                   emb_dict=loc_name_dict, name=name,colname=colname,
                                   negN=nNegTr, posN=nPosTr, testStep=testStep),
            shuffle=shuffle,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=collate_TrainDatasetLocationRS_salesforce
        )

    # Not used in this version
    # criterion = nn.BCEWithLogitsLoss(reduction='none')
    # criterion = nn.CrossEntropyLoss(reduction='none')
    if args.cos_sim_loss:
        criterion = nn.CosineEmbeddingLoss(0.2, reduce=True, reduction='mean')
        lossType = 'cosine'
    else:
        criterion = softmax_loss
        lossType = 'softmax'

    # se- ception dpn can only use finetuned model from imagenet
    model = getattr(rsmodels, args.model)(feat_comp_dim=102, feat_loc_dim=23,
                                          embedding_num=len(loc_name_dict))  # location_recommend_model_v3

    md_path = Path(str(run_root) + '/' + args.ckpt)
    if md_path.exists():
        print('load weights from md_path')
        load_model(model, md_path)

    ##params::Add here
    # params list[models.parameters()]
    # all_params = list(model.parameters())
    all_params = filter(lambda p: p.requires_grad, model.parameters())

    # apply parallel gpu if available
    # model = torch.nn.DataParallel(model)

    # gpu first
    if use_cuda:
        model = model.cuda()

    # print(model)
    if args.mode == 'train':
        if run_root.exists() and args.clean:
            shutil.rmtree(run_root)
        run_root.mkdir(exist_ok=True, parents=True)
        Path(str(run_root) + '/params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        train_loader = make_loader(df_pair=df_train_pair, name='train')
        valid_loader = make_loader(df_pair=df_valid_pair, name='valid')

        train_kwargs = dict(
            args=args,
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            valid_loader=valid_loader,
            patience=args.patience,
            init_optimizer=lambda params, lr: Adam(params, lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=2e-4),
            use_cuda=use_cuda,
        )

        train(params=all_params, **train_kwargs)

    elif args.mode == 'validate':
        """
        For test set that each company in test set x all locations only.
        Because topks recall can not be calculated.
        But if set topks = [0,0,0], roc_auc can be calculated.
        """
        valid_loader = make_loader(df_pair=df_valid_pair, name='valid', shuffle=False)
        validation(model, criterion, tqdm.tqdm(valid_loader, desc='Validation'),
                   use_cuda=use_cuda, lossType=lossType)

    # elif args.mode == 'predict_test':
    #     """
    #     It will generate a score for each company with all the locations(including companies/locations in training set)
    #     or locations of ww only
    #     """
    #     for ind_city in [0, 1, 2, 3, 4]:
    #         pdcl = pd.read_csv(pjoin(TR_DATA_ROOT, clfile[ind_city]))[[colname['location'], colname['company']]]
    #         pdc = pd.read_csv(pjoin(TR_DATA_ROOT, cfile[ind_city]))[[colname['company']]]
    #         pdc[colname['location']] = 'a'
    #         # in case of multi-mapping
    #         # pdcl = pdcl.groupby('atlas_location_uuid').first().reset_index()
    #         all_loc_name = pdcl[[colname['location']]].groupby(colname['location']).first().reset_index()
    #
    #         if wework_location_only:
    #             loc_feat = pd.read_csv(pjoin(TR_DATA_ROOT, lfile))[[colname['location'], 'is_wework']]
    #             loc_ww = loc_feat[loc_feat['is_wework'] == True]
    #             all_loc_name = \
    #             all_loc_name.merge(loc_ww, on=colname['location'], how='inner', suffixes=['', '_right'])[[colname['location']]]
    #
    #         all_loc_name['key'] = 0
    #         pdc['key'] = 0
    #
    #         testing_pair = pd.merge(pdc, all_loc_name, on='key', how='left',
    #                                 suffixes=['_left', '_right']).reset_index(drop=True)
    #
    #         testing_pair = testing_pair.rename(
    #             columns={colname['location']+'_left': 'groundtruth', colname['location']+'_right': colname['location'] })
    #         testing_pair = testing_pair[[colname['company'], colname['location'], 'groundtruth']]
    #         testing_pair['label'] = (testing_pair[colname['location']] == testing_pair['groundtruth'])
    #         testing_pair = testing_pair[[colname['company'], colname['location'] , 'label']]
    #
    #         valid_loader = make_loader(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, df_pair=testing_pair,
    #                                    emb_dict=loc_name_dict, name='valid', shuffle=False)
    #         print('Predictions for city %d' % ind_city)
    #
    #         if wework_location_only:
    #             pre_name = 'ww_'
    #         else:
    #             pre_name = ''
    #
    #         sampling = True if not args.all else False
    #         predict(model, criterion, tqdm.tqdm(valid_loader, desc='Validation'),
    #                 use_cuda=use_cuda, test_pair=testing_pair[[colname['location'], colname['company']]],
    #                 pre_name=pre_name, \
    #                 save_name=pred_save_name[ind_city], lossType=lossType,sampling=sampling)
    #
    # elif args.mode == 'predict_salesforce':
    #     """
    #     It will generate a score for each company with all the locations(including companies/locations in training set)
    #     or locations of ww only
    #     """
    #     pdc_all = pd.read_csv(pjoin(TR_DATA_ROOT, c_salesforce_file))[[colname['company'], 'city']]
    #     for ind_city, str_city in enumerate(cityname):
    #         pdcl = pd.read_csv(pjoin(TR_DATA_ROOT, clfile[ind_city]))[[colname['location'], colname['company']]]
    #         pdc = pdc_all[pdc_all['city'] == str_city]
    #         tot_comp = len(pdc)
    #         print('Total %d company found in %s from salesforce' % (tot_comp, str_city))
    #         pdc[colname['location']] = 'a'
    #         # in case of multi-mapping
    #         # pdcl = pdcl.groupby('atlas_location_uuid').first().reset_index()
    #         all_loc_name = pdcl[[colname['location']]].groupby(colname['location']).first().reset_index()
    #
    #         if wework_location_only:
    #             loc_feat = pd.read_csv(pjoin(TR_DATA_ROOT, lfile))[[colname['location'], 'is_wework']]
    #             loc_ww = loc_feat[loc_feat['is_wework'] == True]
    #             all_loc_name = \
    #                 all_loc_name.merge(loc_ww, on=colname['location'], how='inner', suffixes=['', '_right'])[
    #                     [colname['location']]]
    #
    #         tot_loc = len(all_loc_name)
    #         print('Total %d locations belonged to ww in %s' % (tot_loc, str_city))
    #
    #         all_loc_name['key'] = 0
    #         pdc['key'] = 0
    #
    #         testing_pair = pd.merge(pdc, all_loc_name, on='key', how='left',
    #                                 suffixes=['_left', '_right']).reset_index(drop=True)
    #
    #         testing_pair = testing_pair.rename(
    #             columns={colname['location']+'_left': 'groundtruth', colname['location']+'_right': colname['location']})
    #         testing_pair = testing_pair[[colname['company'], colname['location'], 'groundtruth']]
    #         testing_pair['label'] = (testing_pair[colname['location']] == testing_pair['groundtruth'])
    #         testing_pair = testing_pair[[colname['company'], colname['location'], 'label']]
    #
    #         valid_loader = make_loader(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, df_pair=testing_pair,
    #                                    emb_dict=loc_name_dict, name='valid', shuffle=False)
    #         print('Predictions for city %d' % ind_city)
    #
    #         if wework_location_only:
    #             pre_name = 'ww_'
    #         else:
    #             pre_name = ''
    #
    #         if args.query_location:
    #             topk = min(50, tot_comp)
    #         else:
    #             topk = min(3, tot_loc)
    #
    #         sampling = True if not args.all else False
    #         predict(model, criterion, tqdm.tqdm(valid_loader, desc='Validation'),
    #                 use_cuda=use_cuda, test_pair=testing_pair[[colname['location'], colname['company']]],
    #                 pre_name=pre_name, \
    #                 save_name=pred_save_name[ind_city], lossType=lossType, query_loc_flag=args.query_location,
    #                 topk=topk, sampling=sampling)


# =============================================================================================================================
# End of main
# =============================================================================================================================

# =============================================================================================================================
# train
# =============================================================================================================================
def train(args, model: nn.Module, criterion, *, params,
          train_loader, valid_loader, init_optimizer, use_cuda,
          n_epochs=None, patience=2, max_lr_changes=3) -> bool:
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    params = list(params)  # in case params is not a list
    # add params into optimizer
    optimizer = init_optimizer(params, lr)

    # model load/save path
    run_root = Path(args.run_root)

    model_path = Path(str(run_root) + '/' + 'model.pt')

    if model_path.exists():
        print('loading existing weights from model.pt')
        state = load_model(model, model_path)
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
        best_f1 = state['best_f1']
    else:
        epoch = 1
        step = 0
        best_valid_loss = 0.0  # float('inf')
        best_f1 = 0

    lr_changes = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss,
        'best_f1': best_f1
    }, str(model_path))

    save_where = lambda ep, svpath: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss,
        'best_f1': best_f1
    }, str(svpath))

    report_each = 100
    log = run_root.joinpath('train.log').open('at', encoding='utf8')
    valid_losses = []
    valid_f1s = []
    lr_reset_epoch = epoch

    # epoch loop
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        tq = tqdm.tqdm(total=(args.epoch_size or
                              (nPosTr + nNegTr) * len(train_loader) * args.batch_size))

        if epoch >= 20 and epoch % 2 == 0:
            lr = lr * 0.9
            adjust_learning_rate(optimizer, lr)
            print('lr updated to %0.8f' % lr)

        tq.set_description('Epoch %d, lr %0.8f' % (epoch, lr))
        losses = []
        tl = train_loader
        if args.epoch_size:
            tl = islice(tl, args.epoch_size // args.batch_size)
        try:
            mean_loss = 0

            for i, batch_dat in enumerate(tl):  # enumerate() turns tl into index, ele_of_tl
                featComp = batch_dat['feat_comp']
                featLoc = batch_dat['feat_loc']
                featId = batch_dat['feat_id']
                targets = batch_dat['target']

                if use_cuda:
                    featComp, featLoc, targets, featId = featComp.cuda(), featLoc.cuda(), targets.cuda(), featId.cuda()

                # common_feat_comp, common_feat_loc, feat_comp_loc, outputs = model(feat_comp=featComp, feat_loc=featLoc)
                model_output = model(feat_comp=featComp, feat_loc=featLoc, id_loc=featId)
                outputs = model_output['outputs']

                if args.cos_sim_loss:
                    out_comp_feat = model_output['comp_feat']
                    out_loc_feat = model_output['loc_feat']
                    cos_targets = 2 * targets.float() - 1.0
                    loss = criterion(out_comp_feat, out_loc_feat, cos_targets)
                    lossType = 'cosine'
                else:
                    loss = criterion(outputs, targets)
                    lossType = 'softmax'

                batch_size = featComp.size(0)

                (batch_size * loss).backward()
                if (i + 1) % args.step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='%1.3f' % mean_loss)

                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)

            write_event(log, step, loss=mean_loss)
            tq.close()
            print('saving')
            save(epoch + 1)
            print('validation')
            valid_metrics = validation(model, criterion, valid_loader, use_cuda, lossType=lossType)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_top1 = valid_metrics['valid_top1']
            valid_roc = valid_metrics['auc']
            valid_losses.append(valid_loss)

            # tricky
            valid_loss = valid_roc
            if valid_loss > best_valid_loss:  # roc:bigger is better
                best_valid_loss = valid_loss
                shutil.copy(str(model_path), str(run_root) + '/model_loss_best.pt')

        except KeyboardInterrupt:
            tq.close()
            # print('Ctrl+C, saving snapshot')
            # save(epoch)
            # print('done.')

            return False
    return True

# #=============================================================================================================================
# #predict
# #=============================================================================================================================
def predict(
        model: nn.Module, criterion, predict_loader, use_cuda, test_pair, save_name: str, pre_name: str = '', topk=300,
        query_loc_flag=True, lossType='softmax', sampling=True) -> Dict[str, float]:
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for batch_dat in predict_loader:
            featComp = batch_dat['feat_comp']
            featLoc = batch_dat['feat_loc']
            featId = batch_dat['feat_id']
            targets = batch_dat['target']
            all_targets.append(targets)  # torch@cpu
            if use_cuda:
                featComp, featLoc, targets, featId = featComp.cuda(), featLoc.cuda(), targets.cuda(), featId.cuda()
            model_output = model(feat_comp=featComp, feat_loc=featLoc, id_loc=featId)
            outputs = model_output['outputs']

            if lossType == 'softmax':
                loss = softmax_loss(outputs, targets)
                all_predictions.append(outputs)
            else:
                out_comp_feat = model_output['comp_feat']
                out_loc_feat = model_output['loc_feat']
                cos_targets = 2 * targets.float() - 1.0
                loss = criterion(out_comp_feat, out_loc_feat, cos_targets)
                all_predictions.append(model_output['outputs_cos'])

            all_losses.append(loss.data.cpu().numpy())

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)  # list->torch
    print('all_predictions.shape: ')
    print(all_predictions.shape)

    if lossType == 'softmax':
        all_predictions = F.softmax(all_predictions, dim=1)
        all_predictions2 = all_predictions[:, 1].data.cpu().numpy()
    else:
        all_predictions = (all_predictions + 1) / 2  # squeeze to [0,1]
        all_predictions2 = all_predictions.data.cpu().numpy()

    print('saving...')
    dat_pred_pd = pd.DataFrame(data=all_predictions2.reshape(-1, 1), columns=['similarity'])
    res_pd = pd.concat([test_pair, dat_pred_pd], axis=1)

    if sampling:
        print('sampling...')
        # for each location we return topk companies
        if query_loc_flag:
            sample_pd = res_pd.groupby('atlas_location_uuid').apply(
                lambda x: x.nlargest(topk, ['similarity'])).reset_index(drop=True)
        else:
            sample_pd = res_pd.groupby('duns_number').apply(
                lambda x: x.nlargest(topk, ['similarity'])).reset_index(drop=True)
        sample_pd.to_csv(pjoin(TR_DATA_ROOT, 'sampled_' + pre_name + save_name))
    else:
        print('saving total data...')
        res_pd.to_csv(pjoin(TR_DATA_ROOT, 'all_' + pre_name + save_name))

    roc_auc = 0

    metrics = {}
    metrics['valid_f1'] = 0  # fbeta_score(all_targets, all_predictions, beta=1, average='macro')
    metrics['valid_loss'] = np.mean(all_losses)
    metrics['valid_top1'] = 0  # acc[0].item()
    metrics['auc'] = roc_auc
    metrics['valid_top5'] = 0  # acc[1].item()

    print(' | '.join(['%s %1.3f' % (k, v) for k, v in sorted(metrics.items(), key=lambda kv: -kv[1])]))

    return metrics


# =============================================================================================================================
# validation
# =============================================================================================================================
def validation(
        model: nn.Module, criterion, valid_loader, use_cuda, lossType='softmax') -> Dict[str, float]:
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for batch_dat in valid_loader:
            featComp = batch_dat['feat_comp']
            featLoc = batch_dat['feat_loc']
            featId = batch_dat['feat_id']
            targets = batch_dat['target']
            all_targets.append(targets)  # torch@cpu
            if use_cuda:
                featComp, featLoc, targets, featId = featComp.cuda(), featLoc.cuda(), targets.cuda(), featId.cuda()
            model_output = model(feat_comp=featComp, feat_loc=featLoc, id_loc=featId)
            outputs = model_output['outputs']

            if lossType == 'softmax':
                loss = softmax_loss(outputs, targets)
                all_predictions.append(outputs)
            else:
                out_comp_feat = model_output['comp_feat']
                out_loc_feat = model_output['loc_feat']
                cos_targets = 2 * targets.float() - 1.0
                loss = criterion(out_comp_feat, out_loc_feat, cos_targets)
                all_predictions.append(model_output['outputs_cos'])

            all_losses.append(loss.data.cpu().numpy())

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)  # list->torch
    print('all_predictions.shape: ')
    print(all_predictions.shape)

    if lossType == 'softmax':
        all_predictions = F.softmax(all_predictions, dim=1)
        all_predictions2 = all_predictions[:, 1].data.cpu().numpy()
    else:
        all_predictions = (all_predictions + 1) / 2  # squeeze to [0,1]
        all_predictions2 = all_predictions.data.cpu().numpy()

    all_targets = all_targets.data.cpu().numpy()

    # save_obj(all_targets,'all_targets')
    # save_obj(all_predictions2,'all_predictions2')

    fpr, tpr, roc_thresholds = roc_curve(all_targets, all_predictions2)

    roc_auc = auc(fpr, tpr)

    # if topK > 0 and num_loc > 0:
    #     if Query_Company:  # topk accuracy for company query
    #         all_predictions2 = all_predictions2.reshape(-1, num_loc)
    #         all_targets = all_targets.reshape(-1, num_loc)
    #         print('topk data reforming checking: ', (all_targets.sum(axis=1) == 1).all())
    #     else:
    #         all_predictions2 = all_predictions2.reshape(num_loc, -1)
    #         all_targets = all_targets.reshape(num_loc, -1)
    #
    #     topk_recall = topk_recall_score_all(pred=all_predictions2, truth=all_targets, topk=topK)
    #
    #     step = int(topK / 10)
    #     x = list(range(1, topK + 1))
    #     y = list(topk_recall.reshape(-1))
    #     plt.figure()
    #     plt.plot(x, y)
    #     plt.grid()
    #
    #     for z in range(10, topK + 1, step):
    #         z = z - 1
    #         plt.text(z, y[z], '%.4f' % y[z], ha='center', va='bottom', fontsize=9)
    #
    #     plt.xlabel("topk")
    #     plt.ylabel("recall")
    #
    #     if Query_Company:
    #         plt.title("topk recall curve of %d location" % num_loc)
    #         plt.savefig('topk_recall_of_company_query_%d.jpg' % num_loc)
    #     else:
    #         plt.title("topk recall curve of companies with %d locations" % num_loc)
    #         plt.savefig('topk_recall_of_location_query_%d.jpg' % num_loc)
    #     plt.close()

    metrics = {}
    metrics['valid_f1'] = 0  # fbeta_score(all_targets, all_predictions, beta=1, average='macro')
    metrics['valid_loss'] = np.mean(all_losses)
    metrics['valid_top1'] = 0  # acc[0].item()
    metrics['auc'] = roc_auc
    metrics['valid_top5'] = 0  # acc[1].item()

    print(' | '.join(['%s %1.3f' % (k, v) for k, v in sorted(metrics.items(), key=lambda kv: -kv[1])]))

    return metrics


def _reduce_loss(loss):
    return loss.sum() / loss.shape[0]


def softmax_loss(results, labels):
    labels = labels.view(-1)
    loss = F.cross_entropy(results, labels, reduce=True)

    return loss



if __name__ == '__main__':
    main()
