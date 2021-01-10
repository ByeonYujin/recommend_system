# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd

df = pd.read_csv("./data/u.data", sep='\t', header=None)
df.columns = ["user_id", "item_id", "rating", "timestamp"]
df.head()
# -

df.shape

# +
# 데이터 탐색

df.groupby(["rating"])[["user_id"]].count()
# -

df.groupby(["item_id"])[["user_id"]].count().head()

# +
n_users = df.user_id.unique().shape[0] # unique한 user의 수
n_items = df.item_id.unique().shape[0] # unique한 item의 수

n_users, n_items

# +
import numpy as np

ratings = np.zeros((n_users, n_items)) # 0으로 초기화된 n_users X n_items matrics
ratings.shape

# +
for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]
    
type(ratings)
# -

ratings.shape

ratings

# +
# train data와 test data 분리

from sklearn.model_selection import train_test_split

ratings_train, ratings_test = train_test_split(ratings, test_size=0.33, random_state=42)
ratings_train.shape, ratings_test.shape

# +
# 사용자 기반 협업 필터링

from sklearn.metrics.pairwise import cosine_distances

cosine_distances(ratings_train) # 사용자 간 코사인 유사도 행렬
# -

distances = 1 - cosine_distances(ratings_train)
distances

distances.shape # 정방행렬

# 평가 예측
user_pred = distances.dot(ratings_train) / np.array([np.abs(distances).sum(axis=1)]).T

# +
# 모델 성능 측정
from sklearn.metrics import mean_squared_error

def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


# -

np.sqrt(get_mse(user_pred, ratings_train)) # train data

np.sqrt(get_mse(user_pred, ratings_test)) # test data

# +
# 가장 비슷한 k명을 찾는 비지도 방식의 이웃 검색

from sklearn.neighbors import NearestNeighbors

k = 5
neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
# -

neigh.fit(ratings_train)

top_k_distances, top_k_users = neigh.kneighbors(ratings_train, return_distance=True)

top_k_distances.shape, top_k_users.shape

top_k_users

top_k_distances

# +
# 선택된 k명의 사용자들의 평가 가중치 합을 사용한 예측 및 모델의 성능 측정

user_pred_k = np.zeros(ratings_train.shape)

for i in range(ratings_train.shape[0]):
    user_pred_k[i, :] = top_k_distances[i].T.dot(ratings_train[top_k_users][i]) / np.array([np.abs(top_k_distances[i].T).sum(axis=0)]).T
# -

user_pred_k.shape

user_pred_k

# 모델 평가
np.sqrt(get_mse(user_pred_k, ratings_train))

np.sqrt(get_mse(user_pred_k, ratings_test))


