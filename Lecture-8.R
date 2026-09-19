# DATA STRUCTURES IN R

# 1. VECTOR

# Creating a numeric vector
age <- c(20, 21, 22, 23, 24)
age

# Character vector
name <- c("Rahim", "Karim", "Jannat", "Nadia")

# Logical vector
passed <- c(TRUE, TRUE, FALSE, TRUE)

# Sequence vector using : operator
x <- 1:10

# Sequence vector using seq() function
x <- seq(1,20,2)

# Repeating values using rep()
x <- rep(5,4)

# Repeating character patterns
x <- rep(c("Male","Female"), 3)

# 2. VECTOR OPERATIONS

# Defining input vectors
x <- c(10, 20, 30, 40)
y <- c(2, 4, 6, 8)

# Element-wise arithmetic operations
x + y
x - y
x * y
x / y

# Summary statistical functions
sum(x)
mean(x)
median(x)
min(x)
max(x)
sd(x)
length(x)

# 3. MATRIX

# Creating a matrix (default column-wise filling)
m <- matrix(
  c(10, 20, 30, 40, 50, 60),
  nrow = 2,
  ncol = 3
)
m

# Creating a matrix filled by row
m <- matrix(
  c(10, 20, 30, 40, 50, 60),
  nrow = 2,
  byrow = TRUE
)
m

# Checking matrix dimensions and structure
nrow(m)
ncol(m)
dim(m)
length(m)

# 4. MATRIX INDEXING

# Creating a 3x3 matrix for indexing demonstration
m <- matrix(
  1:9,
  nrow = 3,
  byrow = TRUE
)

# Accessing specific elements and subsets [row, column]
m[2, 3]       # Single element at Row 2, Column 3
m[2, ]        # Entire 2nd row
m[, 3]        # Entire 3rd column
m[1:2, 2:3]   # Submatrix: Rows 1-2, Columns 2-3

# 5. ARRAY

# Creating a 3-D array (3 rows, 4 columns, 2 layers)
arr <- array(
  1:24,
  dim = c(3, 4, 2)
)
arr

# Array indexing [row, column, layer]
arr[1, 2, 1]  # Element at Row 1, Column 2, Layer 1
arr[, , 1]     # Entire Layer 1


# 6. FACTOR

# Creating a categorical vector and converting it to a factor
gender <- c("Male", "Female", "Female", "Male", "Female")
gender_factor <- factor(gender)
gender_factor

# Inspecting factor structure
levels(gender_factor)
nlevels(gender_factor)
table(gender_factor)

# Creating an ordered factor (ordinal variable)
grade <- c("A", "B", "C", "A", "B")
grade <- factor(
  grade,
  levels = c("C", "B", "A"),
  ordered = TRUE
)

# 7. LIST

# Creating a list containing mixed data types
student <- list(
  Name = "Rahim",
  Age = 20,
  Marks = c(80, 85, 90),
  Passed = TRUE
)
student

# Accessing list elements using different syntax
student$Name     # Access by element name
student[[2]]     # Access value of 2nd element
student[1]       # Returns sub-list containing 1st element
student[[1]]     # Returns actual content of 1st element

# 8. DATA FRAME

# Creating a tabular data structure (Data Frame)
student <- data.frame(
  StudentID = c(101, 102, 103, 104),
  Name = c("Rahim", "Karim", "Jannat", "Nadia"),
  Department=c("Statistics","Economics","Statistics","Statistics"),
  Gender = c("Male", "Male", "Female", "Female"),
  Age = c(20, 21, 19, 22),
  Marks = c(85, 92, 78, 88)
)
student

# 9. DATA FRAME FUNCTIONS

# Inspecting properties of the data frame
nrow(student)
ncol(student)
dim(student)
names(student)
str(student)
summary(student)

# 10. DATA FRAME INDEXING

# Slicing data frame rows and columns [row, column]
student[2, 3]        # Row 2, Column 3
student[2, ]         # Row 2 (all columns)
student[, 3]         # Column 3 (all rows)
student[1:3, ]       # Rows 1 to 3
student[, 2:4]       # Columns 2 to 4
student[c(1, 3), ]   # Specific rows 1 and 3

# 11. SELECTING COLUMNS USING $

# Extracting column vectors using $ symbol
student$Name
student$Marks

# Performing calculations on specific columns
mean(student$Marks)

# 12. LOGICAL INDEXING

# Filtering rows based on single condition
student[student$Marks > 80, ]

# Filtering rows based on multiple conditions (AND logic)
student[student$Marks > 80 & student$Age < 22,]

# Filtering rows based on multiple conditions (OR logic)
student[student$Department == "Statistics" | student$Department == "Economics",]

# 13. GROUPED DATA (USING DPLYR)

# Load required library for data manipulation
library(dplyr)

# Grouping data and calculating mean mark per department
student %>%group_by(Department) %>%summarise(Mean_Marks = mean(Marks, na.rm = TRUE))

# 14. MULTIPLE GROUPED STATISTICS

# Calculating multiple summary metrics per department
student %>%
  group_by(Department) %>%
  summarise(
    Number = n(),
    Mean_Marks = mean(Marks, na.rm = TRUE),
    Maximum = max(Marks, na.rm = TRUE),
    Minimum = min(Marks, na.rm = TRUE)
  )

# 15. GROUPING BY GENDER

# Grouping data by Gender to calculate counts and averages
student %>%
  group_by(Gender) %>%
  summarise(
    Number = n(),
    Mean_Marks = mean(Marks, na.rm = TRUE)
  )

# 16. GROUPING BY TWO VARIABLES

# Multi-level grouping (by Department and Gender)
student %>%
  group_by(Department, Gender) %>%
  summarise(
    Mean_Marks = mean(Marks, na.rm = TRUE),
    Number = n()
  )

# 17. FINDING HELP IN R

# Accessing R documentation for specific functions
?mean
help(mean)

# Searching for topics across all installed documentation
??regression

# Getting help for matrix structure
?matrix

# 18. Practical Lab Exercise Dataset Setup
student <- data.frame(
  StudentID = c(101, 102, 103, 104, 105, 106),
  Name = c("Rahim", "Karim", "Jannat", "Nadia", "Sakib", "Mim"),
  Gender = c("Male", "Male", "Female", "Female", "Male", "Female"),
  Department = c("Statistics", "Statistics", "Economics", "Computer Science", "Statistics", "Economics"),
  Age = c(20, 21, 19, 22, 20, 21),
  Marks = c(85, 92, 78, 88, 65, 95)
)

# Task 1: Create a vector containing all students' Marks.
marks_vector <- student$Marks
marks_vector

# Task 2: Calculate the mean and standard deviation of Marks.
mean_marks <- mean(student$Marks)
sd_marks <- sd(student$Marks)
mean_marks
sd_marks

# Task 3: Create a matrix containing Age and Marks.
age_marks_matrix <- as.matrix(student[, c("Age", "Marks")])
age_marks_matrix

# Task 4: Create a factor for Gender.
gender_factor <- factor(student$Gender)
gender_factor

# Task 5: Display the levels of Gender.
levels(gender_factor)

# Task 6: Create a list containing Name, Age, and Marks.
student_list <- list(
  Name = student$Name,
  Age = student$Age,
  Marks = student$Marks
)
student_list

# Task 7: Extract the third element of the list.
student_list[[3]]

# Task 8: Display the first three rows of the data frame.
student[1:3, ]

# Task 9: Display only the Name and Marks columns.
student[, c("Name", "Marks")]

# Task 10: Display students with Marks > 80.
student[student$Marks > 80, ]

# Task 11: Display students from Statistics.
student[student$Department == "Statistics", ]

# Task 12: Calculate the mean Marks by Department.
library(dplyr)
student %>%
  group_by(Department) %>%
  summarise(Mean_Marks = mean(Marks, na.rm = TRUE))

# Task 13: Calculate the number of students by Gender.
student %>%
  group_by(Gender) %>%
  summarise(Number = n())

# Task 14: Calculate the mean Marks by Department and Gender.
student %>%
  group_by(Department, Gender) %>%
  summarise(Mean_Marks = mean(Marks, na.rm = TRUE))

