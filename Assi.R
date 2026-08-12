student=read.csv("C://Users//Asus//Downloads//student_dataset_50rows.csv", header=T)
student
student1=read.delim("C://Users//Asus//Downloads//student_dataset_50rows.txt", header=T)
student1
install.packages("readxl")
library(readxl)
student2=read_excel("C:/Users/Asus/Downloads/student_dataset_50rows.xlsx")
student2

# 1. First 10 observations
head(student,10)

# 2. Last 10 observations
tail(student,10)

# 3. Structure of the dataset
str(student)

# 4. Summary statistics
summary(student)

# 5. Number of rows and columns
nrow(student)
ncol(student)
dim(student)

# 6. Variable names
names(student)

# 7. Data type of each variable
sapply(student,class)

# 8. Total number of missing values
sum(is.na(student))

# 9. Missing values per variable
colSums(is.na(student))

# 10. Remove all missing values
student_clean <- na.omit(student)
student_clean

# 11. Create Bonus_Marks = Marks + 5
library(dplyr)
student_clean<-student_clean%>%mutate(Bonus_Marks=Marks+5)
student_clean

# 12. Create Performance category based on Marks
student_clean<-student_clean%>%mutate(Performance=case_when(Marks >=90 ~ "Excellent",Marks >= 80 ~ "Very Good",Marks >= 70 ~ "Good",Marks>=60~"Average",TRUE ~ "Poor"))
student_clean

# 13. Rename Marks -> FinalMarks
student_clean<-student_clean%>%rename(FinalMarks=Marks)
names(student_clean)

# 14. Add a new student record
new_student <- data.frame(
  StudentID   = 151,
  Name        = "Tanvir",
  Gender      = "Male",
  Department  = "Computer Science",
  Age         = 21,
  FinalMarks  = 88,
  Bonus_Marks = 93,
  Performance = "Very Good"
)
students=rbind(student_clean,new_student)
students

# 15. Students whose names start with "R"
students[startsWith(students$Name, "R"), ]

# 16. Students whose age is between 20 and 22
students[students$Age >= 20 & students$Age <= 22, ]

# 17. Count of male and female students
table(students$Gender)

# 18. Change all department names to uppercase
students$Department <- toupper(students$Department)
students

# 19. Create AgeGroup column
students<-students%>%mutate(AgeGroup=case_when(Age >22 ~ "Adult",Age >= 20 ~ "Young Teen",TRUE ~ "Teen"))                           
students                          

# 20. Remove duplicate records if any exist
duplicated(students)
student[duplicated(students),]
students <- unique(students)
students

# Save the final cleaned dataset
write.csv(students, "C://Users//Asus//Downloads//students.csv")
