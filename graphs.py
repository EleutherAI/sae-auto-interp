# %% 

import torch
import matplotlib.pyplot as plt

def model_one_tokens(n):
    input_tokens = 4000 * n 
    output_tokens = 300 * n
    return input_tokens + output_tokens

def model_two_tokens(n):
    input_tokens = 700 + (n * 12)
    output_tokens = 2 + (n * 8)
    return input_tokens + output_tokens

def generate_data(start, end, num_points, token_func):
    n_values = torch.linspace(start, end, num_points)
    tokens = torch.tensor([token_func(n.item()) for n in n_values])
    return n_values, tokens

def plot_token_growth(data_dict, title="Token Usage Comparison", xlabel="Number of Inputs (N)", ylabel="Total Tokens"):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for label, (x, y) in data_dict.items():
        ax.plot(x, y, label=label)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# Generate data for both models
start, end, num_points = 0, 20, 20  # Adjust range as needed

model1_x, model1_y = generate_data(start, end, num_points, model_one_tokens)
model2_x, model2_y = generate_data(start, end, num_points, model_two_tokens)

# Create a dictionary with the data
data_dict = {
    "Simulator": (model1_x, model1_y),
    "Classifier": (model2_x, model2_y)
}

# Plot the data
plot_token_growth(data_dict)