import numpy as np
from lightfm import LightFM
import csv
from scipy.sparse import coo_matrix

#fetch data and format
def fetch_and_format_data():
    #matrix creation

    data, row_col = [], []

    #Streamers and Users
    streamers, users = {}, {}
    #download at https://cseweb.ucsd.edu/~jmcauley/datasets.html#twitch and place 100k_a.csv in data folder
    with open("data/100k_a.csv") as f:
        for n, line in enumerate(f):
            if n == 10000: break
            t_data = line.split(',')
            user = t_data[0]
            streamer_name = t_data[2]
            streamer_time = int(t_data[4])-int(t_data[3])

            if user not in users:
                users[user] = len(users)
            
            if streamer_name not in streamers:
                streamers[streamer_name] = {
                    "name": streamer_name,
                    "id" : len(streamers)
                }
            if ([users[user], streamers[streamer_name]['id']] in row_col):
                data[row_col.index([users[user], streamers[streamer_name]['id']])] += streamer_time
            else:
                data.append(streamer_time)
                row_col.append([users[user], streamers[streamer_name]['id']])
    row, col = zip(*row_col)
    coo = coo_matrix((data, (row,col)), shape=(len(row), len(col)))

    dictionary = {
        "matrix" : coo,
        "streamers" : streamers,
        "users" : len(users)
        }
    
    return dictionary

data = fetch_and_format_data()

#create model
model = LightFM(loss="warp")

#train model
model.fit(data["matrix"], epochs = 30, num_threads = 2)

def sample_reccomendation(model, coo_m, user_ids):

    #num of users and movies
    n_items = coo_m.shape[1]

    #generate recs for users
    for user_id in user_ids:

        #streamers they already like
        known_pos = coo_m.tocsr()[user_id].indices

        #streamers our model predicts
        scores = model.predict(user_id, np.arange(n_items))
        #ranks the streamers
        top_items = np.argsort(-scores)

        #print out the results
        print("user %s"%user_id)
        print("     Top Known Positives:")


        for x in known_pos.tolist()[:10]:
            for streamer, values in data['streamers'].items():
                if int(x) == values['id']:
                    print ('         - %s' % values['name'])
        print("     Top 10 Recommended:")

        for x in top_items.tolist()[:10]:
            for streamer, values in data['streamers'].items():
                if int(x) == values['id']:
                    print ('         - %s' % values['name'])

sample_reccomendation(model, data["matrix"], [3, 15, 230])