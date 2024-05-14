import matplotlib.pyplot as plt

# Data
episodes = [i for i in range(1, 452, 25)]
performance = [18.8, 9.4, 54.2, 40.6, 20.0, 53.0, 177.4, 108.6, 106.2, 188.4, 
               293.4, 293.0, 385.6, 487.4, 354.4, 433.8, 433.6, 364.2, 405.4]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(episodes, performance, marker='o', linestyle='-', color='b')
plt.title('Performance over 1000 Episodes')
plt.xlabel('Episode')
plt.ylabel('Performance')
plt.grid(True)
plt.tight_layout()

# Mark best performances
best_indices = [i for i, score in enumerate(performance) if score == max(performance)]
for index in best_indices:
    plt.annotate('Best', xy=(episodes[index], performance[index]), xytext=(-15, 10), 
                 textcoords='offset points', arrowprops=dict(arrowstyle='->', color='red'))

plt.show()
