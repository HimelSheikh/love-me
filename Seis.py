#111111111111111111111111111111111111111111
#Loop Examples in Python
#1. Loop through a List
fruits = ['apple', 'banana', 'cherry']
for fruit in fruits:
    print(fruit)
#2. Loop with range()
for i in range(5):
    print(i)
#3. While loop
i = 0
while i < 5:
    print(i)
    i += 1
#4. Loop through a String
for char in 'hello':
    print(char)
#5. Break in a For Loop
for i in range(10):
    if i == 3:
        break
print(i)
#6. Continue in a For Loop
for i in range(5):
 if i == 2:
     continue
 print(i)
#7. While loop with break
i = 0
while True:
    if i == 4:
        break
    print(i)
    i += 1

#8. While loop with continue
i = 0
while i < 5:
    i += 1
    if i == 3:
        continue
    print(i)
#9. For-else Loop
for i in range(3):
    print(i)
else:
    print('Loop completed.')
#10. Nested For Loops
for i in range(1, 4):
    for j in range(1, 4):
        print(f'{i} * {j} = {i * j}')
#11. Using enumerate()
colors = ['red', 'green', 'blue']
for index, color in enumerate(colors):
    print(index, color)
#12. Loop through Dictionary
person = {'name': 'Alice', 'age': 25}
for key, value in person.items():
    print(f'{key}: {value}')
#13. Using zip()
names = ['Anna', 'Ben']
scores = [90, 85]
for name, score in zip(names, scores):
    print(f'{name} scored {score}')
#14. Loop in Reverse
for i in reversed(range(5)):
    print(i)

#15. Loop with sorted()
nums = [3, 1, 4, 2]
for n in sorted(nums):
    print(n)
#16. List Comprehension (Squares)
squares = [x**2 for x in range(5)]
print(squares)
#17. Conditional List Comprehension
evens = [x for x in range(10) if x % 2 == 0]
print(evens)
#18. Set Comprehension
unique_letters = {letter for letter in 'banana'}
print(unique_letters)
#19. Dictionary Comprehension
cubes = {x: x**3 for x in range(4)}
print(cubes)
#20. Count Vowels in Text
text = 'Hello World'
vowels = 'aeiouAEIOU'
for ch in text:
    if ch in vowels:
        print(f'Vowel found: {ch}')


#222222222222222222222222222222222222222222222222
#Question: Create a program that reminds the user to drink water 5 times a day using a loop.
# Program 1: Drink Water Reminder (5 times)
for reminder in range(1, 6):
    print(f"Reminder {reminder}: Drink Water")

#Question: Remind user to drink water every 2 hours from 6 AM to 6 PM.
# Program 2: Drink Water Reminder between 6 AM and 6 PM
hour = 6
for reminder in range(1, 7):
    print(f"Reminder {reminder} at {hour}:00 - Drink Water")
    hour += 2

#Question: Display a message one character at a time (like typing).
# Program 3: Typing Animation
message = "Welcome to the Game!"
for ch in message:
    print(ch)

#Question: Stop searching when “milk” is found in the inventory.
# Program 4: Item Search in Warehouse

inventory = ["bread", "eggs", "milk", "butter"]

for item in inventory:
    print(f"Checking: {item}")
    if item == "milk":
        print("✅ Item found:", item)
        break

#Question: Display all available items, skip the one marked ‘unavailable’.
# Program 5: Skip Unavailable Items
items = ["laptop", "mouse", "unavailable", "keyboard", "monitor"]

print("Available items:")
for item in items:
    if item == "unavailable":
        continue
    print("-", item)

#Question: Simulate traffic light waiting until it turns green.
# Program 6: Traffic Light Countdown
for second in range(1, 6):
    print(f"Red Light... {second} second(s)")
print("✅ Green Light!")

#Question: Simulate elevator stopping from 1–20, skipping floor 13.
# Program 7: Elevator skipping 13th floor
for floor in range(1, 21):
    if floor == 13:
        continue
    print(f"Elevator stopping at floor {floor}")

#Question: Countdown from 4 to 0 for a rocket launch.
# Program 8: Rocket Launch Countdown
for count in range(4, -1, -1):
    print(count)
print("Lift Off!")




#3333333333333333333333333333333333333333333333333333
def fun(name):
    print(f"Hi, {name}")


fun("Ali")


def equation(x, y):
    return (x * 2 + y * 2 + 2 * x * y)


equation(3, 4)
equation(-3, 2)


def sq(x):
    return x * x


sq(5)


def cal(x, y):
    return x + y, x * y


cal(10, 15)


def rect_a_p(length, width):
    area = length * width
    perimeter = 2 * (length + width)
    return area, perimeter


rect_a_p(2, 6)

length = 5
width = 2
area, perimeter = rect_a_p(length, width)
print(f"area : {area}")
print(f"perimeter : {perimeter}")

import math


def area_circle(radius):
    return math.pi * radius * 2


area_circle(3)


def even(n):
    return n % 2 == 0


print(even(4))
print(even(7))


def m(a, b):
    return a if a > b else b


print(m(7, 12))


def fact(n):
    if n == 0:
        return 1
    return n * fact(n - 1)


print(fact(5))


def count_vowels(text):
    vowels = "aeiouAEIOU"
    return sum(1 for char in text if char in vowels)


print(count_vowels("Statistics and Data Science"))

print(count_vowels("Jahangirnagar University"))


def rev_string(st):
    return st[::-1]


print(rev_string("python"))


def count_words(sentence):
    return len(sentence.split())


print(count_words("More para in the Dept of SDS"))


def count_charecter(text, ch):
    return text.count(ch)


print(count_charecter("mango", "a"))
print(count_charecter("Jahangirnagar", "a"))


def sum_list(lst):
    return sum(lst)


print(sum_list([1, 2, 3, 4, 5]))


def largest(lst):
    return max(lst)


print(largest([4, 10, 20, 50, 34]))


def ave(nums):
    return sum(nums) / len(nums)


print(ave([10, 20, 30]))


def check_number(n):
    if n > 0:
        return "Positive"
    elif n < 0:
        return "Negaive"
    else:
        return "Zero"


print(check_number(-5))
print(check_number(5))
print(check_number(0))



#4444444444444444444444444444444444444444444444444444444
import numpy as np
data=np.array([5,15,10,20,25,30,25,35])

#Mean
mean=np.mean(data)
print('Mean:',mean)

#Median
median=np.median(data)
print('Median:',median)

#Mode
from scipy import stats
mode=stats.mode(data)
print("Mode:",mode[0])

#Show dictionary
dir(np)
import scipy as sc
dir(sc)

#SD
std_dev=np.std(data)
print("Standard Deviation:",std_dev)

#Variance
variance=np.var(data)
print('Variance',variance)

#Skewnwess ,Kurtosis
from scipy import stats
Skewness=stats.skew(data)
Kurtosis=stats.kurtosis(data)
print('Skewness:',Skewness)
print('Kurtosis:',Kurtosis)

X=np.array([1,2,3,4,5])
Y=np.array([2,4,5,4,5])
correlation_matrix=np.corrcoef(X,Y)
correlation= correlation_matrix[0,1]
print("Pearson correlation coefficient:",correlation)
mean_X= np.mean(X)
mean_Y= np.mean(Y)




#5555555555555555555555555555555555555555555555555555555555555555
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




#66666666666666666666666666666666666666666666666666666666
#####Mean by Class
import pandas as pd

# Sample data
data = {
    'Class': ['A', 'A', 'A', 'B', 'B', 'C', 'C'],
    'Student': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7'],
    'Score': [85, 90, 88, 75, 78, 92, 95]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

df['Mean by Group'] = df.groupby('Class')['Score'].transform(lambda x: x.mean())
print("\nMean within each class:\n", df)


######Standard deviation by Class
import pandas as pd

# Sample data
data = {
    'Class': ['A', 'A', 'A', 'B', 'B', 'C', 'C'],
    'Student': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7'],
    'Score': [85, 90, 88, 75, 78, 92, 95]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

df['Standard deviation by Group'] = df.groupby('Class')['Score'].transform(lambda x: x.std())
print("\nSD within each class:\n", df)



######Mean and Standard deviation by Class
import pandas as pd

# Sample data
data = {
    'Class': ['A', 'A', 'A', 'B', 'B', 'C', 'C'],
    'Student': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7'],
    'Score': [85, 90, 88, 75, 78, 92, 95]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

df['Standard deviation by Group'] = df.groupby('Class')['Score'].transform(lambda x: x.std())
df['Mean by Class'] = df.groupby('Class')['Score'].transform(lambda x: x.mean())
print("\nSD and Mean within each class:\n", df)



######Mean by Class and Student Section
import pandas as pd


data = {
    'Class': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C'],
    'section': ['S1', 'S2','S1', 'S2', 'S3', 'S4', 'S5', 'S3', 'S4','S5', 'S7','S7'],
    'Score': [85, 90, 88, 75, 78, 92, 95, 85, 90, 88, 75, 78]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

df['Mean by Class and section'] = df.groupby(['Class', 'section'])['Score'].transform(lambda x: x.mean())
print("\nSD within each class:\n", df)



##### Contigency table
import pandas as pd

# Sample categorical data
data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Female', 'Male'],
    'Class': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'A', 'B', 'B'],
    'Passed': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)
print(" Sample Data:\n", df)

contingency_table = pd.crosstab([df['Class'], df['Gender']], df['Passed'], margins=True)

print("\n Contingency Table (Class + Gender vs Passed):\n", contingency_table)




#### Pivot table
import pandas as pd

data = {
    'Class': ['A', 'A', 'B', 'B', 'A', 'B'],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female'],
    'Score': [85, 90, 88, 75, 78, 92]
}

df = pd.DataFrame(data)

pivot = pd.pivot_table(df, values='Score', index='Class', columns='Gender', aggfunc='sum')
print("Pivot Table:\n", pivot)

##### Cross tab
crosstab = pd.crosstab(df['Gender'], df['Class'])
print("\nCross-Tabulation:\n", crosstab)



data2 = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female'],
    'Passed': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}
df2 = pd.DataFrame(data2)

contingency = pd.crosstab(df2['Gender'], df2['Passed'])
print("\nContingency Table:\n", contingency)



###### Chi_square table
import pandas as pd
from scipy.stats import chi2_contingency

# Data
data2 = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female','Female', 'Male', 'Male', 'Female'],
    'Passed': ['Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No','No','Yes','No']
}
df2 = pd.DataFrame(data2)

# Step 1: Create contingency table
contingency = pd.crosstab(df2['Gender'], df2['Passed'])
print("\n Contingency Table:\n", contingency)

# Step 2: Apply Chi-Square test
chi2, p, dof, expected = chi2_contingency(contingency)

# Step 3: Print results
print("\n Chi-Square Test Results:")
print(f"Chi-square statistic = {chi2:.4f}")
print(f"Degrees of freedom = {dof}")
print(f"P-value = {p:.4f}")
print("\nExpected Frequencies:\n", pd.DataFrame(expected, index=contingency.index, columns=contingency.columns))


row_percent = contingency.div(contingency.sum(axis=1), axis=0) * 100
print("\nRow-wise Percentage (%):\n", row_percent.round(2))

# Assuming 'contingency' is your contingency table
column_percent = contingency.div(contingency.sum(axis=0), axis=1) * 100

# Print the column-wise percentage, rounded to 2 decimal places
print("\nColumn-wise Percentage (%):\n", column_percent.round(2))

combined = contingency.astype(str) + " (" + row_percent.round(1).astype(str) + "%" ")"
print(" Contingency Table with Row-wise Percentages:\n", )
print(combined)




#7777777777777777777777777777777777777777777777777777777777
#Q-1
#(i) Calculate the mean score for each class
import numpy as np

# Scores
class_one = np.array([85, 90, 88])
class_three = np.array([75, 78])
class_five = np.array([92, 95])

# Mean for each class
mean_one = np.mean(class_one)
mean_three = np.mean(class_three)
mean_five = np.mean(class_five)

print("Mean Scores:")
print(f"Class One: {mean_one:.2f}")
print(f"Class Three: {mean_three:.2f}")
print(f"Class Five: {mean_five:.2f}")

#(ii) Compare the average scores to determine the best performing class
# Compare means
mean_dict = {"Class One": mean_one, "Class Three": mean_three, "Class Five": mean_five}
best_class = max(mean_dict, key=mean_dict.get)

print("\nAverage Score Comparison:")
for k, v in mean_dict.items():
    print(f"{k}: {v:.2f}")

print(f"\nThe best performing class is: {best_class}")

#(iii) Estimate standard deviation and coefficient of variation
# Standard deviation (sample SD) for each class
std_one = np.std(class_one, ddof=1)
std_three = np.std(class_three, ddof=1)
std_five = np.std(class_five, ddof=1)

# Coefficient of Variation (CV = SD / Mean * 100)
cv_one = (std_one / mean_one) * 100
cv_three = (std_three / mean_three) * 100
cv_five = (std_five / mean_five) * 100

print("\nStandard Deviation and Coefficient of Variation:")
print(f"Class One  → SD = {std_one:.2f},  CV = {cv_one:.2f}%")
print(f"Class Three → SD = {std_three:.2f},  CV = {cv_three:.2f}%")
print(f"Class Five → SD = {std_five:.2f},  CV = {cv_five:.2f}%")

#(iv) Estimate mean score and standard deviation at a time (summary table)
import pandas as pd

# Create summary DataFrame
results = pd.DataFrame({
    "Class": ["One", "Three", "Five"],
    "Mean": [mean_one, mean_three, mean_five],
    "Standard Deviation": [std_one, std_three, std_five],
    "Coefficient of Variation (%)": [cv_one, cv_three, cv_five]
})

print("\nSummary Table:\n")
print(results.round(2))

#Q-2
#Organize the data
# Data: Scores of student sections in different classes
classes = {
    "Class A": {
        "S1": [85, 88],
        "S2": [90, 75],
        "S4": [90],
        "S5": [88]
    },
    "Class B": {
        "S3": [78, 85],
        "S4": [92],
        "S5": [95]
    },
    "Class C": {
        "S7": [75, 78]
    }
}

#i. Calculate average score for each section
# Calculate average scores
average_scores = {}

for class_name, sections in classes.items():
    average_scores[class_name] = {}
    for section, scores in sections.items():
        average_scores[class_name][section] = sum(scores) / len(scores)

# Display results
print("Average Scores for Each Section:")
for class_name, sections in average_scores.items():
    print(f"{class_name}:")
    for section, avg in sections.items():
        print(f"  {section}: {avg:.2f}")

#ii. Calculate standard deviation for each section
import math

# Function to calculate standard deviation
def std_dev(scores):
    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
    return math.sqrt(variance)

# Calculate standard deviation
std_scores = {}

for class_name, sections in classes.items():
    std_scores[class_name] = {}
    for section, scores in sections.items():
        std_scores[class_name][section] = std_dev(scores)

# Display results
print("\nStandard Deviation for Each Section:")
for class_name, sections in std_scores.items():
    print(f"{class_name}:")
    for section, std in sections.items():
        print(f"  {section}: {std:.2f}")

#Q-3
#Organize the data
import pandas as pd

# Data
data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Female', 'Male'],
    'Class': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'A', 'B', 'B'],
    'Passed': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)
print(df)

#Create contingency tables
# Contingency table: Gender vs Passed
gender_table = pd.crosstab(df['Gender'], df['Passed'])
print("\nGender vs Passed:\n", gender_table)

# Contingency table: Class vs Passed
class_table = pd.crosstab(df['Class'], df['Passed'])
print("\nClass vs Passed:\n", class_table)

#Perform Chi-Square Test
from scipy.stats import chi2_contingency

# Chi-square test for Gender vs Passed
chi2_gender, p_gender, dof_gender, expected_gender = chi2_contingency(gender_table)
print(f"\nChi-square test (Gender vs Passed): chi2 = {chi2_gender:.2f}, p-value = {p_gender:.4f}")

# Chi-square test for Class vs Passed
chi2_class, p_class, dof_class, expected_class = chi2_contingency(class_table)
print(f"Chi-square test (Class vs Passed): chi2 = {chi2_class:.2f}, p-value = {p_class:.4f}")


#88888888888888888888888888888888888888888888888888888
#CT-1
#1. Calculate total monthly cost
# Given data
fixed_overhead = 20000
unit_cost = 33.80
total_units = 1000

# Calculate total cost for each unit from 1 to 1000
for units_produced in range(1, total_units + 1):
    total_cost = fixed_overhead + (unit_cost * units_produced)
    print(f"Total cost for {units_produced} units: {total_cost:.2f} taka")
#2. Countdown from 10 to 0 and launch rocket
import time

# Countdown
for i in range(10, -1, -1):
    print(i)
    time.sleep(1)  # Wait 1 second between counts (optional)

print("Rocket Launched!")

#3.Reminder to take a break 4 times between 9 AM and 5 PM
# Break intervals: 9 AM, 11 AM, 1 PM, 3 PM
break_times = ["9:00 AM", "11:00 AM", "1:00 PM", "3:00 PM"]

for time_slot in break_times:
    print(f"Reminder at {time_slot}: Take a Break")

#4. Average of first 20 natural numbers
# First 20 natural numbers
numbers = list(range(1, 21))
average = sum(numbers) / len(numbers)
print(f"Average of first 20 natural numbers: {average}")

#CT-2
#1. Central Tendency for Monthly Income
import numpy as np

# Monthly income data
income = [50, 45, 34, 38, 42, 800, 34, 39, 42, 44]

# Calculate mean, median, and mode
mean_income = np.mean(income)
median_income = np.median(income)

print(f"Mean: {mean_income}")
print(f"Median: {median_income}  <-- Appropriate due to outlier")

#2. Geometric / Poisson Approximation for Bulbs
# Geometric: P(Y <= 5) = 1 - (1-p)^5
n=5
p=0.03
prob_within_5 = 1 - (1 - p)**n
print(f"Probability first defective within 5 bulbs: {prob_within_5:.4f}")

#ii. Poisson-process (Exponential) approximation
import math

prob_approx = 1 - math.exp(-n * p)
print(f"Exponential approximation: {prob_approx:.4f}")

#iii. Expected runs requiring more than 50 inspections
runs = 10000
threshold = 50

prob_more_50 = (1 - p)**threshold
expected_runs = runs * prob_more_50
print(f"Expected runs needing >50 inspections: {expected_runs:.0f}")

#3. Visualize the distribution function
import matplotlib.pyplot as plt
import numpy as np

# x values
x = np.linspace(20, 50, 100)
F_x = (x - 20) / (50 - 20)

# Plot
plt.plot(x, F_x, label='F(x)')
plt.title('Distribution Function F(x)')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid(True)
plt.legend()
plt.show()


