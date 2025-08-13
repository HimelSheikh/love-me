import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [5, 7, 4, 6, 8]

# Create scatter plot
plt.scatter(x, y, color='blue', marker='v', label='Data Points')

# Connect points with a line
plt.plot(x, y, color='red', linestyle='-', linewidth=1, label='Connecting Line')

# Add value labels for each point
for i in range(len(x)):
    plt.text(x[i], y[i], f'({x[i]},{y[i]})', fontsize=9, ha='left', va='top')

# Add labels, title, and legend
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Scatter Plot with Connecting Line')
plt.legend()

# Show plot
plt.show()

#####two scatter line in a plot
import matplotlib.pyplot as plt

# Data
x = [1, 2, 3, 4, 5]
y = [5, 7, 4, 6, 8]
z = [2, 4, 6, 7, 8]

# Plot x vs y
plt.scatter(x, y, color='blue', marker='o', label='x vs y')
plt.plot(x, y, color='blue', linestyle='-', linewidth=1)

# Plot x vs z
plt.scatter(x, z, color='red', marker='^', label='x vs z')
plt.plot(x, z, color='red', linestyle='--', linewidth=1)

# Labels and title
plt.xlabel('x')
plt.ylabel('Values')
plt.title('Scatter Plot with Two Lines')
plt.legend()

plt.show()

###Histogram

import matplotlib.pyplot as plt
import numpy as np
import math

# Sample data
data = [12, 15, 13, 17, 19, 22, 22, 23, 25, 26, 26, 29, 30, 33, 35]

# Calculate bins using Sturges' formula
n = len(data)
bins = math.ceil(np.log2(n) + 1)

# Plot histogram and get counts and bin edges
counts, bin_edges, patches = plt.hist(data, bins=bins, color='skyblue', edgecolor='black')

# Calculate bin midpoints for labeling
bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2


# Add axis labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title(f'Histogram with bin midpoints labeled (bins={bins})')

# Set new x-axis labels
plt.xticks(bin_mids, [f'{mid:.1f}' for mid in bin_mids])
plt.show()




#### Bar Diagram
import matplotlib.pyplot as plt
from collections import Counter

# Your categorical data
data = ['A', 'A', 'B', 'A', 'C', 'A', 'B', 'C', 'D', 'E', 'D', 'F', 'E', 'F', 'C', 'D', 'A', 'F', 'A']

# Count frequency of each category
freq = Counter(data)

# Separate the categories and their counts
categories = list(freq.keys())
values = list(freq.values())

# Create bar plot
plt.bar(categories, values, color='skyblue', edgecolor='red')

# Add labels and title
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.title('Bar Diagram of Categorical Data')

# Show plot
plt.show()

#### Pie chart
import matplotlib.pyplot as plt
from collections import Counter

# Your categorical data
data = ['A', 'A', 'B', 'A', 'C', 'A', 'B', 'C', 'D', 'E', 'D', 'F', 'E', 'F', 'C', 'D', 'A', 'F', 'A']

# Count frequency of each category
freq = Counter(data)

# Separate the categories and their counts
labels = list(freq.keys())
sizes = list(freq.values())

# Create pie chart
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel1.colors)

# Equal aspect ratio ensures pie is a circle
plt.axis('equal')
plt.title('Pie Chart of Categorical Data')

# Show plot
plt.show()


import matplotlib.pyplot as plt

# Data
x = [1, 2, 3, 4, 5]
y = [5, 7, 4, 6, 8]
z = [2, 4, 6, 7, 8]

# Plot x vs y
plt.scatter(x, y, color='blue', marker='o', label='x vs y')
plt.plot(x, y, color='blue', linestyle='-', linewidth=1)

# Plot x vs z
plt.scatter(x, z, color='red', marker='^', label='x vs z')
plt.plot(x, z, color='red', linestyle='--', linewidth=1)

# Labels and title
plt.xlabel('x')
plt.ylabel('Values')
plt.title('Scatter Plot with Two Lines')
plt.legend()

plt.show()
