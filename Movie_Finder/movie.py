import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch data and format
data = fetch_movielens(min_rating=4.0)

#Print test and train data
print(repr(data['train']))
print(repr(data['test']))

#create model
model = LightFM(loss="warp")

#train model
model.fit(data["train"], epochs = 30, num_threads = 2)

def sample_reccomendation(model, data, user_ids):

    #num of users and movies
    n_users, n_items = data["train"].shape

    #generate recs for users
    for user_id in user_ids:

        #movies they already like
        known_pos = data["item_labels"][data["train"].tocsr()[user_id].indices]

        #movies our model predicts
        scores = model.predict(user_id, np.arange(n_items))
        #ranks the movies
        top_items = data["item_labels"][np.argsort(-scores)]

        #print out the results
        print("user %s"%user_id)
        print("     Top 10 Known Positives:")


        for x in known_pos[:10]:
            print("         %s" %x)
        print("     Top 10 Recommended:")

        for x in top_items[:10]:
            print("         %s"%x)

sample_reccomendation(model, data, [3, 5, 450])