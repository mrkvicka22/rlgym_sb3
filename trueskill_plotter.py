import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
with open("policy_ratings", "rb") as f:
    ratings_database = pickle.load(f)

order_function = lambda x: int(x.split('_')[2])
x = [order_function(ratings_database['opponents'][i]["name"]) for i in range(len(ratings_database['opponents']))]
y = [ratings_database['opponents'][i]["rating"].mu for i in range(len(ratings_database['opponents']))]
y_lower = [ratings_database['opponents'][i]["rating"].mu - ratings_database['opponents'][i]["rating"].sigma for i in range(len(ratings_database['opponents']))]
y_upper = [ratings_database['opponents'][i]["rating"].mu + ratings_database['opponents'][i]["rating"].sigma for i in range(len(ratings_database['opponents']))]

fig = go.Figure([
    go.Scatter(
        x=x,
        y=y,
        line=dict(color='rgb(0,100,80)'),
        mode='lines'
    ),
    go.Scatter(
        x=x+x[::-1], # x, then x reversed
        y=y_upper+y_lower[::-1], # upper, then lower reversed
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    )
])
fig.show()


plt.plot([order_function(ratings_database['opponents'][i]["name"]) for i in range(len(ratings_database['opponents']))],[ratings_database['opponents'][i]["rating"].mu for i in range(len(ratings_database['opponents']))] )
plt.ylabel('mu')
plt.show()