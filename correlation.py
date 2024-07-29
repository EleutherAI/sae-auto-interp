# %%

explanation_dir = "results/gpt2_explanations"
recall_dir = "results/gpt2_recall"
fuzz_dir = "results/gpt2_fuzz"

import os
import orjson

scores = []

def _score(data):

    score = 0

    for d in data:
        if d['ground_truth'] == d['prediction']:
            score += 1

    return score / len(data)

for file in os.listdir(recall_dir):
    feature_name = file.replace(".txt", "")

    try:

        with open(f"{explanation_dir}/{file}", "rb") as f:
            explanation = orjson.loads(f.read())

        with open(f"{recall_dir}/{file}", "rb") as f:
            recall = orjson.loads(f.read())

        with open(f"{fuzz_dir}/{file}", "rb") as f:
            fuzz = orjson.loads(f.read())

        scores.append({
            "explanation": explanation['explanation'],
            "recall": _score(recall),
            "fuzz": _score(fuzz)
        })

    except Exception as e:
        continue

# %%
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from scipy import stats

# Assuming 'scores' is your list of dictionaries containing the data

# Extract recall and fuzz scores
recall_scores = np.array([item['recall'] for item in scores])
fuzz_scores = np.array([item['fuzz'] for item in scores])
explanations = [item['explanation'] for item in scores]

# Calculate correlation coefficient and R-squared
correlation, p_value = stats.pearsonr(recall_scores, fuzz_scores)
r_squared = correlation ** 2

# Create linear regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(recall_scores, fuzz_scores)
line = slope * recall_scores + intercept

# Create the scatter plot
fig = go.Figure()

# Add scatter plot
fig.add_trace(go.Scatter(
    x=recall_scores,
    y=fuzz_scores,
    mode='markers',
    # name='Data points',
    marker=dict(
        size=10,
        color=fuzz_scores,
        colorscale='Viridis',
        showscale=True
    ),
    text=explanations,
    hoverinfo='text+x+y'
))

# Add correlation line
fig.add_trace(go.Scatter(
    x=recall_scores,
    y=line,
    mode='lines',
    # name='Correlation Line',
    line=dict(color='red', dash='dash')
))

# Update the layout
fig.update_layout(
    title='Correlation between Recall and Fuzz Scores',
    xaxis_title='Recall Score',
    yaxis_title='Fuzz Score',
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1]),
    width=800,
    height=600,
    annotations=[
        dict(
            x=0.05,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"Correlation: {correlation:.3f}<br>R²: {r_squared:.3f}",
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
    ]
)

# Show the plot
fig.show()