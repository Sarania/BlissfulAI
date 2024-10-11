import numpy as np
import matplotlib.pyplot as plt

# Define the weight function based on recency (using your formula i**0.7)
def weight_function_recency(num_entries):
    return [i**0.7 for i in range(1, num_entries + 1)]

# Define upvote and downvote modifiers
upvote_modifier = 3
downvote_modifier = 0.25

# Generate a sample dataset of recency weights
num_memories = 100  # Let's assume we have 100 memories to visualize
recency_weights = weight_function_recency(num_memories)

# Simulate user ratings (randomly generating + and -)
np.random.seed(42)  # Seed for reproducibility
ratings = np.random.choice(['+', '-', ''], size=num_memories, p=[0.2, 0.1, 0.7])  # 20% upvote, 10% downvote, 70% no rating

# Apply the rating modifiers to the recency weights
adjusted_weights = [
    weight * upvote_modifier if rating == '+' else
    weight * downvote_modifier if rating == '-' else
    weight
    for weight, rating in zip(recency_weights, ratings)
]

# Plot the weights before and after applying user ratings
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_memories + 1), recency_weights, label="Recency Weights", linestyle='dotted')
plt.plot(range(1, num_memories + 1), adjusted_weights, label="Adjusted Weights (Recency + User Ratings)", color='orange')
plt.scatter(range(1, num_memories + 1), adjusted_weights, c=['green' if r == '+' else 'red' if r == '-' else 'gray' for r in ratings], label='Rating Modifiers')

# Labels and legend
plt.xlabel("Memory Index (1 = Most Recent, 100 = Least Recent)")
plt.ylabel("Weight")
plt.title("Effect of Recency and User Ratings on Memory Selection Weights")
plt.legend()
plt.grid(True)

# Show plot
plt.show()
