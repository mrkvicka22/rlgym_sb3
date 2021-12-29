import pickle
import matplotlib.pyplot as plt
with open("policy_ratings", "rb") as f:
    ratings_database = pickle.load(f)

plt.plot([int(ratings_database['opponents'][i]["name"].split("_")[1]) for i in range(len(ratings_database['opponents']))],[ratings_database['opponents'][i]["rating"].mu for i in range(len(ratings_database['opponents']))] )
plt.ylabel('mu')
plt.show()