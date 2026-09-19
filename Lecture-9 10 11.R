# 1. VECTORS IN R

# Method 1: Using c() (Combine Function)
numbers <- c(10, 20, 30, 40, 50)

# Creating Different Types of Vectors
x <- c(2.5, 4.8, 7.1) # Numeric Vector
y <- c(2L, 4L, 6L) # Integer Vector
fruits <- c("Apple", "Banana", "Orange") # Character Vector
passed <- c(TRUE, FALSE, TRUE, TRUE) # Logical Vector

# Creating Sequences
1:10 # Ascending sequence 1 to 10
10:1 # Descending sequence 10 to 1
seq(1, 10, 2) # Sequence with step size 2
seq(0, 1, by = 0.2) # Sequence from 0 to 1 with step size 0.2

# Using rep() to repeat values
rep(5, 4) # Repeat 5 four times
rep(c(1, 2, 3), 3) # Repeat pattern (1, 2, 3) three times

# Naming Vector Elements
marks <- c(80, 75, 90)
names(marks) <- c("Alice", "Bob", "Charlie")
marks

# Indexing Vectors
x <- c(5, 10, 15, 20)
x[1] # First element
x[3] # Third element
x[c(2, 4)] # Multiple elements
x[2:4] # Range of elements
x[length(x)] # Last element

# Negative Indexing
x[-2] # Exclude second element
x[-c(1, 4)] # Remove first and fourth elements

# Logical Indexing
marks <- c(55, 72, 88, 39, 91)
marks > 60 # Returns logical vector
marks[marks > 60] # Extract values matching condition

# Modifying Vector Elements
x <- c(10, 20, 30)
x[2] <- 100
x

# Missing Values in Vectors
x <- c(10, 20, NA, 40)
mean(x) # Returns NA
mean(x, na.rm = TRUE) # Mean ignoring missing values

marks <- c(75, 82, 68, 90, 77)
mean(marks)
max(marks)
marks[marks >= 80]

# 2. READING AND WRITING DATA IN R

# Reading CSV Files
student <- read.csv("Student.csv")
View(student)

read.csv(file = "Student.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)

# Reading CSV using readr Package
install.packages("readr")
library(readr)
student <- read_csv("Student.csv")

# Reading Excel Files
install.packages("readxl")
library(readxl)
student <- read_excel("Student.xlsx")
student <- read_excel("Student.xlsx", sheet = "Marks")
student <- read_excel("Student.xlsx", sheet = 2)

# Reading Text Files
student <- read.table("Student.txt", header = TRUE)
student <- read.delim("Student.txt")

# Reading SPSS Files
install.packages("haven")
library(haven)
survey <- read_sav("BDHS2022.sav")

# Exploring Imported Data
head(student) # First six rows
tail(student) # Last six rows
str(student) # Structure of dataset
summary(student) # Summary statistics
dim(student) # Dimensions (rows, columns)
names(student) # List variable names
sapply(student, class) # Check data types

# Exporting Data
write.csv(student, "Student New.csv", row.names = FALSE)

install.packages("writexl")
library(writexl)
write_xlsx(student, "Student.xlsx")

write.table(student, "Student.txt", sep = "\t", row.names = FALSE)

# 3. DATA MANIPULATION & CLEANING

# Creating a Dataset
student <- data.frame(
  StudentID = c(101, 102, 103, 104, 105, 106, 107, 108),
  Name = c("Rahim", "Karim", "Jannat", "Nadia", "Sakib", "Mim", "Arif", "Suma"),
  Gender = c("Male", "Male", "Female", "Female", "Male", "Female", "Male", "Female"),
  Department = c("Statistics", "Statistics", "Economics", "Computer Science", "Statistics", "Economics", "Mathematics", "Statistics"),
  Age = c(20, 21, 19, 22, 20, 21, 23, 20),
  Marks = c(85, 92, 78, 88, 65, 95, 75, 81)
)
student

# Inspecting Data
View(student)
head(student)
tail(student)
str(student)
summary(student)
dim(student)
names(student)

# Removing Duplicate Records
duplicated(student)
student[duplicated(student), ]
student <- unique(student)

# Renaming & Mutating Variables using dplyr
library(dplyr)

student <- student %>%rename(FinalMarks = Marks)

student <- student %>%mutate(Result = ifelse(FinalMarks >= 40, "Pass", "Fail"))

student <- student %>%
  mutate(Grade = case_when(
    FinalMarks >= 80 ~ "A+",
    FinalMarks >= 70 ~ "A",
    FinalMarks >= 60 ~ "B",
    TRUE ~ "F"
  ))

# Data Subsetting
student$Name
student[, 2]
student[, c("Name", "Department", "FinalMarks")]

student %>% select(Name, Department, FinalMarks)
student %>% select(-Gender)

# Selecting Rows
student[1:5, ]
student[3, ]
student[nrow(student), ]
student[sample(nrow(student), 3), ]

# Selecting Rows and Columns Together
student[1:4, c("Name", "FinalMarks")]

# Filtering Data
student %>% filter(FinalMarks > 80)
student %>% filter(Gender == "Female")
student %>% filter(Department == "Statistics")
student %>% filter(Gender == "Female", FinalMarks > 80)
student %>% filter(Gender == "Female" | FinalMarks > 90)
student %>% filter(Department %in% c("Statistics", "Economics"))

# Sorting Data
student %>% arrange(FinalMarks)
student %>% arrange(desc(FinalMarks))
student %>% arrange(Department, desc(FinalMarks))

# Selecting After Filtering
student %>%
  filter(FinalMarks >= 80) %>%
  select(Name, Department, FinalMarks)

# Pipe Operator Demonstration
student %>%
  filter(Department == "Statistics") %>%
  arrange(desc(FinalMarks)) %>%
  select(Name, FinalMarks)

# 4. HANDLING MISSING DATA

# Creating Example Dataset with NA
student <- data.frame(
  StudentID = c(101, 102, 103, 104, 105, 106),
  Name = c("Rahim", "Karim", "Jannat", "Nadia", "Sakib", "Mim"),
  Age = c(20, 21, NA, 22, 20, NA),
  Marks = c(85, NA, 90, 88, 75, 92),
  Gender = c("Male", "Male", "Female", "Female", "Male", "Female")
)
student

# Identifying Missing Values
is.na(student)
is.na(student$Marks)
sum(is.na(student))
colSums(is.na(student))
rowSums(is.na(student))
student[!complete.cases(student), ]

# Removing Missing Values
student_clean <- na.omit(student)
student_clean <- student[complete.cases(student), ]

# Imputation Techniques
# Mean Imputation
student$Age[is.na(student$Age)] <- mean(student$Age, na.rm = TRUE)
student$Marks[is.na(student$Marks)] <- mean(student$Marks, na.rm = TRUE)

# Median Imputation
student$Age[is.na(student$Age)] <- median(student$Age, na.rm = TRUE)

# Mode Imputation
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
student$Gender[is.na(student$Gender)] <- Mode(student$Gender)

# Multiple Imputation (Advanced)
library(mice)
imp <- mice(student, m = 5)
completed_data <- complete(imp)

sum(is.na(student))

# 5. ASSIGNMENT DATA FRAME PRACTICE

# Load dplyr for data manipulation functions
library(dplyr)

# Create the initial dataset
student <- data.frame(
  StudentID = c(101, 102, 103, 104, 105, 106, 107, 108, 109, 110),
  Name = c("Rahim", "Karim", "Jannat", "Nadia", "Sakib", "Mim", "Arif", "Suma", "Rafi", "Lima"),
  Gender = c("Male", "Male", "Female", "Female", "Male", "Female", "Male", "Female", "Male", "Female"),
  Department = c("Statistics", "Statistics", "Economics", "Computer Science", "Statistics", "Economics", "Mathematics", "Statistics", "Economics", "Computer Science"),
  Age = c(20, 21, NA, 22, 20, 21, 23, 20, 22, NA),
  Marks = c(85, 92, 78, 88, 65, 95, 75, 81, NA, 89)
)
  # 1. Display the dataset.
  student
  
  # 2. Show the first six observations.
  head(student)
  
  # 3. Display the structure of the dataset.
  str(student)
  
  # 4. Count rows and columns.
  dim(student)
  
  # 5. Rename Marks to FinalMarks.
  student <- student %>% 
    rename(FinalMarks = Marks)
  
  # 6. Create Result and Grade variables.
  student <- student %>% 
    mutate(
      Result = ifelse(FinalMarks >= 40, "Pass", "Fail"),
      Grade = case_when(
        FinalMarks >= 80 ~ "A+",
        FinalMarks >= 70 ~ "A",
        FinalMarks >= 60 ~ "B",
        TRUE ~ "F"
      )
    )
  
  # 7. Display students scoring at least 80 marks.
  student %>% filter(FinalMarks >= 80)
  
  # 8. Display female students.
  student %>% filter(Gender == "Female")
  
  # 9. Display Statistics students scoring above 80.
  student %>% filter(Department == "Statistics", FinalMarks > 80)
  
  # 10. Sort students by marks in descending order.
  student %>% arrange(desc(FinalMarks))
  
  # 11. Count missing values by column.
  colSums(is.na(student))
  
  # 12. Replace missing ages with the mean age.
  student$Age[is.na(student$Age)] <- mean(student$Age, na.rm = TRUE)
  
  # 13. Replace missing marks with the median marks.
  student$FinalMarks[is.na(student$FinalMarks)] <- median(student$FinalMarks, na.rm = TRUE)
  
  # 14. Remove duplicate records (if any).
  student <- unique(student)
  
  # 15. Save the cleaned dataset
  write.csv(student, "Student_Cleaned.csv", row.names = FALSE)


# 6. ADDITIONAL DATA STRUCTURES

# Lists in R
student_list <- list(
  name = "Rahim",
  age = 22,
  marks = c(75, 80, 85),
  passed = TRUE
)
student_list

# Accessing List Elements
student_list$name
student_list$age
student_list[[1]]
student_list[1] # Returns a sublist

student_list$department <- "Statistics" # Add element
student_list$passed <- NULL # Remove element

# Arrays in R
x <- array(1:10) # 1D Array
x <- array(1:6, dim = c(2, 3)) # 2D Array (2x3)
x <- array(1:24, dim = c(3, 4, 2)) # 3D Array (3x4x2)

# Access array elements
x[1, 2, 1]
x[2, 3, 1]
x[1, 1, 2]

# Logical & %in% Vector Indexing
x <- c(10, 20, 30, 40, 50)
x[x > 25]
x[x >= 20 & x <= 40]
x[x %in% c(20, 40)]

# Data Frame Filtering / Indexing Examples
student_df <- data.frame(
  ID = 1:5,
  Name = c("Rahim", "Karim", "Hasan", "Nila", "Mina"),
  Gender = c("M", "M", "M", "F", "F"),
  Marks = c(75, 82, 68, 90, 85)
)

student_df$Name
student_df[, "Name"]
student_df[1, ]
student_df[1:3, ]
student_df[1:3, 2:4]
student_df[student_df$Marks > 80, ]
student_df[student_df$Gender == "F", ]
student_df[student_df$Gender == "F" & student_df$Marks > 80, ]

# 7. GROUPING DATA IN R

student <- data.frame(
  Name = c("Rahim", "Karim", "Hasan", "Nila", "Mina", "Rita"),
  Gender = c("M", "M", "M", "F", "F", "F"),
  Department = c("Statistics", "Statistics", "Math", "Statistics", "Math", "Math"),
  Marks = c(75, 82, 68, 90, 85, 78)
)

# Using tapply()
tapply(student$Marks, student$Gender, mean)
tapply(student$Marks, student$Gender, max)
tapply(student$Marks, student$Gender, length)

# Using aggregate()
aggregate(Marks ~ Gender, data = student, FUN = mean)
aggregate(Marks ~ Department, data = student, FUN = mean)
aggregate(Marks ~ Department, data = student, FUN = summary)

# Grouping Using dplyr
student %>% group_by(Gender)

student %>%
  group_by(Gender) %>%
  summarise(Mean_Marks = mean(Marks))

student %>%
  group_by(Gender) %>%
  summarise(
    Number = n(),
    Mean = mean(Marks),
    Minimum = min(Marks),
    Maximum = max(Marks)
  )

student %>%
  group_by(Department) %>%
  summarise(
    Number = n(),
    Mean_Marks = mean(Marks)
  )

student %>%
  group_by(Department, Gender) %>%
  summarise(
    Number = n(),
    Mean_Marks = mean(Marks)
  )

student %>%
  group_by(Gender) %>%
  summarise(Mean_Marks = mean(Marks)) %>%
  filter(Mean_Marks > 80)

# 8. FACTORS IN R

gender <- factor(c("Male", "Female", "Female", "Male", "Male"))
gender
is.factor(gender)
class(gender)
levels(gender)
nlevels(gender)

# Education Level Factors
education <- factor(c("Primary", "Secondary", "Graduate", "Graduate", "Primary"))
education

education <- factor(
  c("Primary", "Secondary", "Graduate"),
  levels = c("Primary", "Secondary", "Graduate")
)

# Ordered Factors
risk <- factor(
  c("Low", "High", "Medium", "Low"),
  levels = c("Low", "Medium", "High"),
  ordered = TRUE
)
risk
is.ordered(risk)
risk[1] < risk[2] # Comparing categories

# Factors in Data Frames
student <- data.frame(
  ID = 1:6,
  Name = c("Rahim", "Karim", "Nila", "Sadia", "Hasan", "Rafi"),
  Gender = c("Male", "Male", "Female", "Female", "Male", "Male"),
  Department = c("Statistics", "Mathematics", "Statistics", "Physics", "Mathematics", "Statistics")
)

student$Gender <- factor(student$Gender)
student$Department <- factor(student$Department)
str(student)

student$Gender <- factor(student$Gender, levels = c("Female", "Male"))

table(student$Gender)
table(student$Gender, student$Department)
summary(student$Gender)

# Changing Factor Levels and Labels
gender <- factor(c("M", "F", "F", "M", "M"))
levels(gender) <- c("Female", "Male")

gender <- factor(gender, levels = c("F", "M"), labels = c("Female", "Male"))

gender <- factor(c("M", "F", "F", "M"), levels = c("F", "M"), labels = c("Female", "Male"))

# Conversions
x <- c("Male", "Female", "Male")
class(x)
x <- factor(x)
class(x)

gender <- factor(c("Male", "Female", "Male"))
gender_character <- as.character(gender)
class(gender_character)

x <- factor(c("10", "20", "30", "40"))
as.numeric(x) # Level codes
as.numeric(as.character(x)) # Actual numbers

education <- c("1", "2", "3", "1", "2")
education <- factor(
  education,
  levels = c("1", "2", "3"),
  labels = c("Primary", "Secondary", "Higher")
)

# 9. CONDITIONAL STATEMENTS


age <- 20
if (age >= 18) {
  print("Adult")
}

# Comparison Operators & Logical Operators
marks <- 40
if (marks >= 40) {
  print("Pass")
}

age <- 25
age >= 18 & age <= 60

marks <- 75
if (marks >= 40 & marks <= 100) {
  print("Valid marks")
}

age <- 65
if (age < 18 | age > 60) {
  print("Special age group")
}

x <- 10
!(x > 5)

# if with multiple commands
marks <- 85
if (marks >= 80) {
  print("Excellent")
  print("Grade A+")
  print("High performance")
}

# if-else Statement
marks <- 35
if (marks >= 40) {
  print("Pass")
} else {
  print("Fail")
}

# Voting Eligibility
age <- 19
if (age >= 18) {
  print("Eligible to vote")
} else {
  print("Not eligible to vote")
}

# Even or Odd
number <- 15
if (number %% 2 == 0) {
  print("Even")
} else {
  print("Odd")
}

# else if Statement - Grade Classification
marks <- 76
if (marks >= 80) {
  grade <- "A+"
} else if (marks >= 70) {
  grade <- "A"
} else if (marks >= 60) {
  grade <- "B"
} else if (marks >= 50) {
  grade <- "C"
} else if (marks >= 40) {
  grade <- "D"
} else {
  grade <- "F"
}
print(grade)

# Nested Conditions
age <- 25
gender <- "Female"
if (age >= 18) {
  if (gender == "Female") {
    print("Adult female")
  } else {
    print("Adult male")
  }
} else {
  print("Minor")
}

# Nested Conditions - Student Performance
marks <- 85
if (marks >= 40) {
  print("Pass")
  if (marks >= 80) {
    print("Distinction")
  }
} else {
  print("Fail")
}

# Nested Conditions with Attendance
marks <- 85
attendance <- 80
if (attendance >= 75) {
  print("Eligible for examination")
  if (marks >= 80) {
    print("Excellent performance")
  } else {
    print("Regular performance")
  }
} else {
  print("Not eligible for examination")
}

# Nested Conditions in Health Data
age <- 65
bp <- 150
if (age >= 60) {
  if (bp >= 140) {
    print("High risk")
  } else {
    print("Moderate risk")
  }
} else {
  print("Lower-risk age group")
}

# BMI Classification
bmi <- 27
if (bmi < 18.5) {
  category <- "Underweight"
} else if (bmi < 25) {
  category <- "Normal"
} else if (bmi < 30) {
  category <- "Overweight"
} else {
  category <- "Obese"
}
print(category)

# Income Classification
income <- 45000
if (income < 20000) {
  group <- "Low"
} else if (income < 50000) {
  group <- "Middle"
} else {
  group <- "High"
}
print(group)

# Scholarship Eligibility
marks <- 85
attendance <- 80
if (marks >= 80 & attendance >= 75) {
  print("Eligible for scholarship")
} else {
  print("Not eligible for scholarship")
}

# Patient Risk
age <- 65
bp <- 150
if (age >= 60 & bp >= 140) {
  risk <- "High"
} else if (age >= 60 | bp >= 140) {
  risk <- "Moderate"
} else {
  risk <- "Low"
}
print(risk)

# Vectorized Conditional Statements - ifelse()
status <- c("Yes", "No", "Yes", "No", "Yes")
status_binary <- ifelse(status == "Yes", 1, 0)
status_binary

age <- c(12, 17, 25, 35, 62, 70)
age_group <- ifelse(age < 18, "Child", ifelse(age < 60, "Adult", "Older Adult"))
age_group

marks <- c(35, 45, 67, 80, 30)
result <- ifelse(marks >= 40, "Pass", "Fail")
result

# case_when()
library(dplyr)
marks <- c(35, 48, 65, 75, 90)
grade <- case_when(
  marks >= 80 ~ "A+",
  marks >= 70 ~ "A",
  marks >= 60 ~ "B",
  marks >= 40 ~ "C",
  TRUE ~ "F"
)
grade

# 10. LOOPING IN R


# For Loop
for (i in 1:5) {
  print(i)
}

for (i in 1:10) {
  print(i)
}

# Print Even Numbers
for (i in 1:20) {
  if (i %% 2 == 0) {
    print(i)
  }
}

# Print Odd Numbers
for (i in 1:20) {
  if (i %% 2 != 0) {
    print(i)
  }
}

# For Loop Through Vectors
students <- c("Rahim", "Karim", "Sumaiya", "Nusrat")
for (name in students) {
  print(name)
}

subjects <- c("Statistics", "R", "Python", "Database")
for (subject in subjects) {
  print(subject)
}

for (i in seq(2, 10, by = 2)) {
  print(i)
}

# Calculations in Loop
for (i in 1:5) {
  square <- i^2
  print(square)
}

# Calculate Sum & Mean
x <- c(10, 20, 30, 40, 50)
total <- 0
for (i in x) {
  total <- total + i
}
print(total)

total <- 0
for (i in x) {
  total <- total + i
}
mean_value <- total / length(x)
print(mean_value)

# Loop with Index
x <- c(10, 20, 30, 40, 50)
for (i in 1:length(x)) {
  print(x[i])
}

# Loops and Conditions
marks <- c(85, 72, 48, 91, 65)
for (mark in marks) {
  if (mark >= 80) {
    print("A")
  } else if (mark >= 60) {
    print("B")
  } else {
    print("C")
  }
}

# Nested For Loop
for (i in 1:3) {
  for (j in 1:3) {
    print(paste(i, j))
  }
}

# Multiplication Table
for (i in 1:5) {
  for (j in 1:10) {
    print(paste(i, "x", j, "=", i * j))
  }
}

# Pass/Fail Iteration
marks <- c(55, 67, 72, 81, 90)
for (mark in marks) {
  if (mark >= 40) {
    print("Pass")
  } else {
    print("Fail")
  }
}

# Count Even Values
x <- c(12, 15, 18, 21, 24, 27)
count <- 0
for (i in x) {
  if (i %% 2 == 0) {
    count <- count + 1
  }
}
print(count)

# While Loop
i <- 1
while (i <= 5) {
  print(i)
  i <- i + 1
}

i <- 2
while (i <= 20) {
  print(i)
  i <- i + 2
}

i <- 1
total <- 0
while (i <= 10) {
  total <- total + i
  i <- i + 1
}
print(total)

i <- 1
total <- 0
while (total <= 100) {
  total <- total + i
  i <- i + 1
}
print(i - 1)
print(total)

# Repeat Loop
i <- 1
repeat {
  print(i)
  i <- i + 1
  if (i > 5) {
    break
  }
}

i <- 1
total <- 0
repeat {
  total <- total + i
  i <- i + 1
  if (i > 10) {
    break
  }
}
print(total)

# Break and Next
for (i in 1:10) {
  if (i == 6) {
    break
  }
  print(i)
}

for (i in 1:10) {
  if (i == 5) {
    next
  }
  print(i)
}

# 11. APPLY FAMILY FUNCTIONS

# apply()
x <- matrix(1:12, nrow = 3)
x
apply(x, 1, sum) # Row sums
apply(x, 2, sum) # Column sums
apply(x, 1, mean) # Row means
apply(x, 2, mean) # Column means
apply(x, 1, min)
apply(x, 1, max)
apply(x, 1, median)
apply(x, 1, sd)

# lapply()
x <- list(a = 1:5, b = 10:15, c = 20:25)
lapply(x, mean)

x <- list(a = 1:3, b = 4:6, c = 7:9)
lapply(x, function(z) z^2)

data <- data.frame(
  age = c(20, 25, 30, 35),
  income = c(20000, 25000, 30000, 40000),
  score = c(70, 80, 75, 90)
)
lapply(data, mean)

# tapply()
marks <- c(70, 80, 65, 90, 75, 85)
group <- c("A", "A", "B", "B", "A", "B")
tapply(marks, group, mean)

age <- c(20, 25, 30, 35, 40, 45)
gender <- c("Male", "Female", "Male", "Female", "Male", "Female")
tapply(age, gender, mean)

region <- c("Urban", "Urban", "Rural", "Rural", "Urban", "Rural")
tapply(age, list(gender, region), mean)

# 12. DATA VISUALIZATION

# Scatter Plots
x <- c(1, 2, 3, 4, 5)
y <- c(2, 5, 4, 8, 10)
plot(x, y,
     main = "Scatter Plot",
     xlab = "X values",
     ylab = "Y values",
     pch = 19
)

df <- data.frame(
  height = c(150, 160, 165, 170, 180),
  weight = c(50, 58, 62, 70, 80)
)
plot(df$height, df$weight,
     main = "Height vs Weight",
     xlab = "Height",
     ylab = "Weight",
     col = "blue",
     pch = 19
)

library(ggplot2)
ggplot(df, aes(x = height, y = weight)) +
  geom_point() +
  labs(title = "Height vs Weight", x = "Height", y = "Weight") +
  theme_minimal()

x <- 1:10
y <- c(3, 5, 4, 7, 8, 10, 9, 12, 11, 14)
group <- c("A", "A", "A", "A", "A", "B", "B", "B", "B", "B")

plot(x, y,
     pch = 19,
     col = ifelse(group == "A", "blue", "red"),
     main = "Scatter Plot by Group",
     xlab = "X",
     ylab = "Y"
)
legend("topleft",
       legend = c("Group A", "Group B"),
       col = c("blue", "red"),
       pch = 19
)

# Line Graphs
v <- c(17, 25, 38, 13, 41)
t <- c(22, 19, 36, 19, 23)
m <- c(25, 14, 16, 34, 29)

plot(v,
     type = "o",
     col = "red",
     xlab = "Month",
     ylab = "Articles Written",
     main = "Articles Written Chart"
)
lines(t, type = "o", col = "blue")
lines(m, type = "o", col = "green")

# Reference Lines - abline()
x <- 1:10
y <- x^2
plot(x, y)
abline(v = 5) # Vertical line
abline(h = 25) # Horizontal line
abline(a = 0, b = 1) # Diagonal line

x2 <- 1:10
y2 <- 2 * x2 + 3
lines(x2, y2, col = "red", lty = 2)

x <- 1:6
y <- c(10, 15, 12, 20, 18, 25)
plot(x, y,
     type = "l",
     main = "Line Graph",
     xlab = "Time",
     ylab = "Value"
)

# Monthly Rainfall Chart
month <- c("Jan", "Feb", "Mar", "Apr", "May", "Jun")
rainfall <- c(12, 18, 35, 80, 220, 310)
plot(rainfall,
     type = "o",
     pch = 19,
     col = "blue",
     xaxt = "n",
     main = "Monthly Rainfall",
     xlab = "Month",
     ylab = "Rainfall (mm)"
)
axis(1, at = 1:6, labels = month)

# Comparing Two Lines
month <- 1:6
rainfall_2025 <- c(12, 18, 35, 80, 220, 310)
rainfall_2026 <- c(15, 25, 40, 95, 200, 280)

plot(month, rainfall_2025,
     type = "o",
     pch = 19,
     col = "blue",
     ylim = c(0, 350),
     xlab = "Month",
     ylab = "Rainfall (mm)",
     main = "Monthly Rainfall Comparison"
)
lines(month, rainfall_2026,
      type = "o",
      pch = 17,
      col = "red"
)
legend("topleft",
       legend = c("2025", "2026"),
       col = c("blue", "red"),
       pch = c(19, 17)
)

# Bar Charts
A <- c(17, 32, 8, 53, 1)
barplot(A, xlab = "X-axis", ylab = "Y-axis", main = "Bar-Chart")

#Customizing the Bar Chart 

students <- c(40, 55, 35, 60)
departments <- c("Statistics", "Botany", "Pharmacy", "Chemistry")

barplot(students,
        names.arg = departments,
        main = "Number of Students by Department",
        xlab = "Department",
        ylab = "Number of Students"
)

#Different color: 

barplot(students,
        names.arg = departments,
        col = c("blue", "red", "green", "orange"),
        main = "Number of Students by Department",
        xlab = "Department",
        ylab = "Number of Students"
)

#Bar Plot from a Frequency Table

sex <- c("Male", "Female", "Female", "Male", "Female", "Female", "Male", "Female")
freq <- table(sex)
barplot(freq,
        col = c("skyblue", "pink"),
        main = "Students by Sex",
        xlab = "Sex",
        ylab = "Frequency"
)

# Text labels on Bar Chart
bp <- barplot(students,
              names.arg = departments,
              col = "skyblue",
              main = "Number of Students",
              ylab = "Students",
              ylim = c(0, 70)
)
text(bp, students, labels = students, pos = 3)

A <- c(17, 2, 8, 13, 1, 22)
B <- c("Jan", "Feb", "Mar", "Apr", "May", "Jun")

barplot(A,
        names.arg = B,
        xlab = "Month",
        ylab = "Articles",
        col = "steelblue",
        main = "GeeksforGeeks - Article Chart",
        cex.main = 1.5,
        cex.lab = 1.2,
        cex.axis = 1.1
)

x <- barplot(A, names.arg = B, col = "steelblue", ylim = c(0, max(A) * 1.2))
text(x = x, y = A + 1, labels = A, pos = 3, cex = 1.2, col = "black")

# Grouped Bar Plot
students <- matrix(c(
  25, 30,
  15, 20,
  20, 25,
  15, 12
), nrow = 2, byrow = TRUE)

colnames(students) <- c("Statistics", "Botany", "Pharmacy", "Chemistry")
rownames(students) <- c("Male", "Female")

# Create grouped bar plot  

bar <- barplot(students,
               beside = TRUE,
               col = c("skyblue", "pink"),
               border = "black",
               main = "Male and Female Students by Department",
               xlab = "Department",
               ylab = "Number of Students",
               ylim = c(0, 35),
               cex.main = 1,
               cex.axis = 0.7,
               cex.lab = 0.7,
               cex.names = 0.7,
               las = 1
)

# Add values above bars 

text(
  x = bar,
  y = students,
  labels = students,
  pos = 3,
  cex = 0.5
)

legend("topright",
       legend = c("Male", "Female"),
       fill = c("skyblue", "pink"),
       border = "black",
       bty = "n",
       cex = 0.5
)

# Plot Grid Configuration
par(mfrow = c(2, 2))
plot(1:5, main = "Plot 1")
plot(5:1, main = "Plot 2")
plot(1:10, main = "Plot 3")
plot(10:1, main = "Plot 4")
