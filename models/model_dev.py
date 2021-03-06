from functools import partial
import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
import math
import torch.utils.model_zoo as model_zoo
from utils import load_model_with_dict_replace, load_model_with_dict, load_model
from collections import OrderedDict
from models.utils import idx_2_one_hot
from models.location_recommendation import setLinearLayer_fast
from torch import cuda

use_cuda = cuda.is_available()
eps = 1e-8


class mi_price_likelihood_v1(nn.Module):
    """
    input: user location
    output: [mu_p,delta_p] parameters of price
    it will pick one of the parameters from the group
    """

    def __init__(self, user_dim, loc_dim, num_groups):
        super().__init__()
        self.user_dim = user_dim
        self.loc_dim = loc_dim
        self.K = num_groups

        self.input_dim = user_dim + loc_dim

        self.user_loc_net = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=self.K),
            nn.Sigmoid(),
        )

        self.theta = nn.Parameter(torch.Tensor(self.K, 2))  # K x [mu,delta]
        self.theta.data.uniform_(-0.1, 0.1)

        self.theta[:, 1] = torch.abs(self.theta[:, 1])

    def forward(self, feat_user, feat_loc, feat_price):
        """

        :param feat_user: 
        :param feat_loc: 
        :param feat_price: [B,1]
        :return: 
        """
        batch_size = feat_user.shape[0]
        price_parameter = self.predict(feat_user, feat_loc)
        mu = price_parameter[:, 0].reshape(-1, 1)  # [B,1]
        delta = price_parameter[:, 1].abs().reshape(-1, 1)  # [B,1]

        ln_delta = delta.log().reshape(-1, 1)  # [B,1]
        p_minus_mu = torch.pow(mu - feat_price, 2)  # [B,1]

        p_minus_mu_div_delta = p_minus_mu / delta.pow(2) / 2  # [B,1]

        loss = ln_delta.sum() - p_minus_mu_div_delta.sum()
        loss /= batch_size
        return -loss

    def predict(self, feat_user, feat_loc):
        feat_user_loc = torch.cat([feat_user, feat_loc], dim=1)
        feat_attention = self.user_loc_net(feat_user_loc)  # [B,K]

        _, max_id = torch.max(feat_attention, dim=1, keepdim=True)  # [B,1],[B,1]

        dum_max_id = idx_2_one_hot(max_id, self.K, use_cuda=use_cuda)  # [B,K]
        dum_max_id = dum_max_id.unsqueeze(dim=2).expand(-1, -1, 2)  # [B,K,2]
        dum_max_id = dum_max_id > 0
        # BGFX
        b_theta = self.theta.expand(dum_max_id.shape[0], -1, -1)  # [B,K,2]

        price_parameter = b_theta[dum_max_id].reshape(-1, 2)  # [B,2]

        return price_parameter


class mi_price_likelihood_v2(nn.Module):
    """
    input: user location
    output: [mu_p,delta_p] parameters of price
    it will pick one of the parameters from the group
    """

    def __init__(self, user_dim, loc_dim, num_groups, cuda_flag=False):
        super().__init__()
        self.user_dim = user_dim
        self.loc_dim = loc_dim
        self.K = num_groups
        self.cuda_flag = cuda_flag

        # self.input_dim = user_dim + loc_dim

        self.user_net = nn.Sequential(
            nn.Linear(in_features=self.user_dim, out_features=32),
            nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.LeakyReLU(),
            nn.Linear(in_features=16, out_features=self.K),
            nn.Sigmoid(),
        )

        self.theta = nn.Parameter(torch.Tensor(self.K, 2, self.loc_dim + 1))  # [K,2,loc_dim+1] *bias is included
        self.theta[:, 0, :].data.uniform_(3, 5)
        self.theta[:, 1, :].data.uniform_(0, 1)

    def forward(self, feat_user, feat_loc, feat_price):
        """

        :param feat_user: 
        :param feat_loc: 
        :param feat_price: [B,1]
        :return: 
        """
        batch_size = feat_user.shape[0]
        price_parameter = self.predict(feat_user, feat_loc)
        mu = price_parameter[:, 0].reshape(-1, 1)  # [B,1]
        delta = price_parameter[:, 1].abs().reshape(-1, 1) + eps  # [B,1]

        ln_delta = delta.log().reshape(-1, 1)  # [B,1]
        p_minus_mu = torch.pow(mu - feat_price, 2)  # [B,1]

        p_minus_mu_div_delta = p_minus_mu / delta.pow(2) / 2  # [B,1]

        loss = ln_delta.sum() - p_minus_mu_div_delta.sum()
        loss /= batch_size
        return -loss

    def predict(self, feat_user, feat_loc):
        """
        price_parameter = price_regression_parameter*feat_loc,
        where price_regression_parameter = f(feat_user),
        f(.) contains K regression parameter.
        By doing so, each user have its own regression parameter.
        :param feat_user: 
        :param feat_loc: 
        :return: price_parameter = [B,2] contains mu and sigma
        """
        feat_attention = self.user_net(feat_user)  # [B,K]

        _, max_id = torch.max(feat_attention, dim=1, keepdim=True)  # [B,1],[B,1]

        dum_max_id = idx_2_one_hot(max_id, self.K, use_cuda=use_cuda)  # [B,K]
        dum_max_id = dum_max_id.unsqueeze(dim=2).expand(-1, -1, 2).unsqueeze(dim=3).expand(-1, -1, -1,
                                                                                           self.loc_dim + 1)  # [B,K]->[B,K,2] -> [B,K,2,loc_dim+1]
        dum_max_id = dum_max_id > 0
        # BGFX
        b_theta = self.theta.expand(dum_max_id.shape[0], -1, -1, -1)  # [K,2,input_dim]-> [B,K,2,loc_dim+1]

        price_regression_parameter = b_theta[dum_max_id].reshape(-1, 2, self.loc_dim + 1)  # [B,2,loc_dim+1]
        one_mat = torch.ones(feat_loc.shape[0], 1)
        if self.cuda_flag:
            one_mat = one_mat.cuda()
        aug_feat_loc = torch.cat([feat_loc, one_mat], dim=1)  # [B,loc_dim+1]

        price_parameter = price_regression_parameter @ aug_feat_loc.unsqueeze(
            dim=2)  # [B,2,loc_dim+1] @ [B,loc_dim+1,1] = [B,2,1]
        price_parameter = price_parameter.squeeze()  # [B,2]
        return price_parameter

    def predict_v2(self, feat_user, feat_loc):
        """
        Replace logical operation by matrix product.
        """
        feat_attention = self.user_net(feat_user)  # [B,K]

        _, max_id = torch.max(feat_attention, dim=1, keepdim=True)  # [B,1],[B,1]

        dum_max_id = idx_2_one_hot(max_id, self.K, use_cuda=use_cuda)  # [B,K]

        dum_max_id = dum_max_id.unsqueeze(dim=1)  # [B,1,K]

        # BGFX
        b_theta = self.theta.expand(dum_max_id.shape[0], -1, -1, -1)  # [K,2,input_dim]-> [B,K,2,loc_dim+1]

        # [B,K,2,loc_dim+1] -> [2,B,K,loc_dim+1]
        b_theta = b_theta.permute(2, 0, 1, 3)

        price_regression_parameter = dum_max_id @ b_theta  # [B,1,K]x[2,B,K,loc_dim+1]--> [2,B,1,loc_dim+1]
        price_regression_parameter = price_regression_parameter.squeeze().permute(1, 0,
                                                                                  2)  # [2,B,loc_dim+1]->[B,2,loc_dim+1]

        one_mat = torch.ones(feat_loc.shape[0], 1)
        if self.cuda_flag:
            one_mat = one_mat.cuda()
        aug_feat_loc = torch.cat([feat_loc, one_mat], dim=1)  # [B,loc_dim+1]
        print('aug_shape:', aug_feat_loc.shape)

        price_parameter = price_regression_parameter @ aug_feat_loc.unsqueeze(
            dim=2)  # [B,2,loc_dim+1] @ [B,loc_dim+1,1] = [B,2,1]
        price_parameter = price_parameter.squeeze()  # [B,2]
        return price_parameter


class mi_price_regression_v1(nn.Module):
    """
    price = f(U)'g(L)
    """
    def __init__(self,user_dim,loc_dim):
        super().__init__()
        self.user_dim = user_dim
        self.loc_dim = loc_dim

        self.loc_emb_dim = 64
        self.loc_net = nn.Sequential(
            nn.Linear(in_features=loc_dim,out_features=self.loc_emb_dim),
            nn.LeakyReLU(),
        )

        self.user_emb_dim = self.loc_emb_dim
        self.user_net = nn.Sequential(
            nn.Linear(in_features=user_dim,out_features=self.user_emb_dim),
            nn.LeakyReLU(),
        )

        self.bias = nn.Parameter(torch.Tensor(1))
        self.bias.data.uniform_(-0.1,0.1)


    def forward(self, feat_user, feat_loc ):
        feat_user_emb = self.user_net(feat_user)
        feat_loc_emb = self.loc_net(feat_loc)
        assert(feat_user_emb.shape == feat_loc_emb.shape)
        feat_price = feat_user_emb*feat_loc_emb #[B,D]
        feat_price = feat_price.sum(dim=1,keepdim=True) + self.bias #[B,1]
        return feat_price

class mi_price_regression_v2(nn.Module):
    """
    price = f(U)'[g(L),h(R)]
    """
    def __init__(self,user_dim,loc_dim,region_dim):
        super().__init__()
        self.user_dim = user_dim
        self.loc_dim = loc_dim
        self.region_dim = region_dim

        self.region_emb_dim = 64
        self.region_net = setLinearLayer_fast(fin=region_dim, fout=self.region_emb_dim)

        self.loc_emb_dim = 64
        self.loc_net = nn.Sequential(
            nn.Linear(in_features=loc_dim,out_features=self.loc_emb_dim),
            nn.LeakyReLU(),
        )

        self.user_emb_dim = self.region_emb_dim + self.loc_emb_dim
        self.user_net = nn.Sequential(
            nn.Linear(in_features=user_dim,out_features=self.user_emb_dim),
            nn.LeakyReLU(),
        )

        self.bias = nn.Parameter(torch.Tensor(1))
        self.bias.data.uniform_(-0.1,0.1)

    def forward(self, feat_user, feat_loc , feat_region):
        """
        feat_region [B,K,region_dim] it can be a batch(B) of K companies inside the region or others.
        feat_loc [B,loc_dim]
        feat_user [B,user_dim]
        """
        feat_user_emb = self.user_net(feat_user) #[B,user_emb_dim]
        feat_loc_emb = self.loc_net(feat_loc) #[B, loc_emb_dim]
        feat_region_emb = self.region_net(feat_region) #[B,region_emb_dim]

        feat_loc_region_emb = torch.cat([feat_loc_emb,feat_region_emb],dim=1) #[B,loc_emb_dim+region_emb_dim]

        assert(feat_loc_region_emb.shape == feat_user_emb.shape )

        feat_price = feat_user_emb*feat_loc_region_emb #[B,D]
        feat_price = feat_price.sum(dim=1,keepdim=True) + self.bias #[B,1]
        return feat_price