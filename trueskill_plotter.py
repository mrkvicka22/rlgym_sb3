import pickle
import matplotlib.pyplot as plt
with open("policy_ratings", "rb") as f:
    ratings_database = pickle.load(f)

order_function = lambda x: int(x.split('_')[2])
plt.plot([order_function(ratings_database['opponents'][i]["name"]) for i in range(len(ratings_database['opponents']))],[ratings_database['opponents'][i]["rating"].mu for i in range(len(ratings_database['opponents']))] )
plt.ylabel('mu')
plt.show()