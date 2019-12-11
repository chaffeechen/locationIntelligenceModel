# Project:

Model part of location intelligence project, including training, validation, predicting and feature importance.

# Implentmented models:

**Wide&Deep[2016]** @ models/location_recommendation.py

The basice model is based on that structure, while the id embedding module is replace by our region model.

# Requirements:

## Must:

``` commandline
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
pip install pretrainedmodels scikit-learn tqdm opencv-python pandas pygeohash
pip install --upgrade scikit-image
```

## Optional
```commandline
pip install imgaug cvxpy cvxopt folium
```

# Projects:

## 1. location recommendation:  main_location_company.py

It build the relationship between location(item) and company(user). 
But each user only buy one item.
It includes train, validation, predict.
Reasoning part is move into main_location_company_model_based_reason.py.

```commandline
nohup python3 -u main_location_company.py --model location_recommend_model_v6 --run_root result/location_recommend_model_v6_191113 --lr 0.01 --mode train --apps _191113.csv >lrm_5c.out 2>lrm_5c.err &
python3 main_location_company.py --model location_recommend_model_v6 --run_root result/location_recommend_model_v6_5city_191113 --lr 0.01 --mode predict_salesforce --ckpt model_loss_best.pt --apps _191113.csv
```

## 2. main_location_company_model_based_reason.py

It generate the feature importance of input layer by b.p. the error of last layer.

```commandline
python3 main_location_company_model_based_reason.py --apps _191113.csv --pre_name sampled_ww_
```

## 3. get_embedding_feature.py

It generate the embedded vector for each location id after the model is trained.

## 4. main_location_intelligence_region.py

It build the relationship between location(item) and company(user). 

* Each user has several item. Because each company inside the circle of a building will be counted in.
* Region model is used to replace id embedding so that it is easy for adding new buildings inside.
* It includes train, validation, predict. 
For predict part, embedded feature of location/region need to be produced before ahead.

```commandline
nohup python3 -u main_location_intelligence_region.py --run_root result/location_RSRBv5_191114 --model location_recommend_region_model_v5 --lr 0.01 --mode train --trainStep 1000 --batch-size 4 --n-epochs 160 >mlir_5.out 2>mlir_5.err &
python3 -u main_location_intelligence_region.py --run_root result/location_RSRBv5_191114 --model location_recommend_region_model_v5 --lr 0.01 --mode validate --trainStep 1000 --batch-size 4 --n-epochs 160
python3 main_location_intelligence_region.py --run_root result/location_RSRBv5_191114 --model location_recommend_region_model_v5 --lr 0.01 --mode predict --batch-size 1 --apps _191114.csv
```

## 5. get_embedding_feature_region.py

It generate the embedded vector for each location id after the model is trained.

```commandline
python3 get_embedding_feature_region.py --path /home/ubuntu/location_recommender_system/ --maxK 100 --model location_recommend_region_model_v5 --run_root result/location_RSRBv5_191114/
```

## 6. main_location_intelligence_region_based_reason.py

It generate the feature importance of input layer by b.p. the error of last layer.
Also, embedded feature of location/region need to be produced before ahead.

```commandline
python3 main_location_intelligence_region_based_reason.py --apps _191114.csv --pre_name sampled_ww_ --run_root result/location_RSRBv5_191114/ --model location_recommend_region_model_v5
```






