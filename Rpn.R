##Lec-2
x=5
y=4
x1=c(2,4,7,9)
x2=c(3,6,5,8)
x
x1
x!=y
x==y
x%%y
x%/%y
sqrt(x)
abs(-x)
round(2.236068,2)
ceiling(2.24)
floor(2.4)
trunc(2.457898)
log(10)
exp(10)
exp(2)
sort(x1)
sort(x2, decreasing=TRUE)
sum(x1)
max(x1)
min(x1)
mean(x1)
sd(x1)
var(x1)
range(x1)
length(x1)
d="statistics"
e="and"
f="Data Science"
paste(d,e,f)
paste0(d,e,f)
nchar(d)
toupper(d)
tolower(d)
substr(d,3,6)
seq(1,10,2)
seq(1,10)
rep(5,7)
runif(10,0,1)
sample(1:10,10)
rnorm(10,0,1)

##Lec-3
x=c(12,13,15,16)
x
x1=c("A","B","C")
x1
x2=c(TRUE, FALSE)
x2
Q=c(16,17,7,20)
names(Q)=c("X","Y","Z","M")
Q
a=1:10
a
a1=10:1
a1
a3=seq(2,20,2)
a3
a4=rep(5,4)
a4
a5=rep(c(1,2,4),4)
a5
a5=x[1]
a5
a7=x[-1]
a7
a6=x[c(2:4)]
a6
b7=x[c(2,4)]
b7

x
x[2]=20
x>10
a8=x[x<16]
a8
x4=c(2,4,6,8,9,NA,5,NA)
m=mean(x4,na.rm=TRUE)
m
d=na.omit(x4)
d
is.na(x4)
!is.na(x4)
!na.omit(x4)
x4


y=c(27,NA,-5,60)
z=na.omit(y)
z
m=z[-2]
m


##Lec-4
d=data(iris)
d
head(iris)
str(iris)
names(iris)
summary(iris)
data("CO2")
names(CO2)
d=read.csv("C://Users//Asus//Downloads//tab.csv", header=T)
head(d)
names(d)
d1=read.delim("C://Users//Asus//Downloads//table.txt", header=T)
head(d1)
install.packages("readxl")
library(readxl)
d2=read_excel("C:/Users/Asus/Downloads/table.xlsx")
d2



##Lec-5
#Built in data
data("AirPassengers")
head(AirPassengers)
head(iris)
str(iris)
names(iris)
summary(iris)
data("CO2")
names(CO2)
####################
# Create a sample student dataset

student <- data.frame(
  StudentID = c(101,102,103,104,105,106,107,108,109,110,
                111,112,113,114,115),
  
  Name = c("Rahim","Karim","Jannat","Nadia","Sakib",
           "Fatema","Rafi","Mim","Arif","Nusrat",
           "Rahim","Karim","Hasan","Rima","Tania"),
  
  Gender = c("Male","Male","Female","Female","Male",
             "Female","Male","Female","Male","Female",
             "male","Male","Male","Female","Female"),
  
  Age = c(20,21,19,22,20,
          NA,21,20,23,22,
          20,21,19,NA,20),
  
  Department = c("Statistics","Statistics","Mathematics",
                 "Economics","Statistics",
                 "Computer Science","Mathematics",
                 "Statistics","Economics",
                 "Computer Science",
                 "Statistics","Statistics",
                 "Mathematics","Economics",
                 "Computer Science"),
  
  Marks = c(85,92,78,88,65,
            90,NA,81,75,95,
            85,92,68,80,NA)
)

student
write.csv(student,
          "C://Users//Asus//Downloads//tablet.csv",
          row.names = FALSE)

write.table(student,
            "C://Users//Asus//Downloads//tablet.txt",
            row.names = FALSE)

install.packages("writexl")
library(writexl)
write_xlsx(student,
           "C://Users//Asus//Downloads//tablet.xlsx")


# View data
View(student)

# First six rows
head(student)

# Last six rows
tail(student)

# Structure
str(student)

# Summary
summary(student)

# Dimensions
dim(student)

# Variable names
names(student)

is.na(student)
sum(is.na(student))
colSums(is.na(student))
rowSums(is.na(student))

duplicated(student)
student[duplicated(student),]
unique(student)



##Lec-6
student <- data.frame(
  StudentID=c(101,102,103,104,105,106,107,108),
  Name=c("Rahim","Karim","Jannat","Nadia",
         "Sakib","Mim","Arif","Suma"),
  Gender=c("Male","Male","Female","Female",
           "Male","Female","Male","Female"),
  Department=c("Statistics","Statistics",
               "Economics","Computer Science",
               "Statistics","Economics",
               "Mathematics","Statistics"),
  Age=c(20,21,19,22,20,21,23,20),
  Marks=c(85,92,78,88,65,95,75,81)
)
head(student)
names(student)
dim(student)

duplicated(student)
student[duplicated(student),]
student <- unique(student)

library(dplyr)
student <- student %>%
  rename(FinalMarks=Marks)
names(student)

student <- student %>%
  mutate(Result=ifelse(FinalMarks>=40,"Pass", "Fail"))
head(student)

student <- student %>%
  mutate(Grade=case_when(FinalMarks>=80~"A", FinalMarks>=70~"B",FinalMarks>=60~"C", TRUE~"F"))
head(student)

student$Name
student[,2]
student[,c(2,4)]
student[,c(1:4)]
student[1:3,]

student %>%filter(FinalMarks>80)
student %>%filter(Gender=="Female")
student %>%filter(Department=="Statistics")
student %>%filter(Gender=="Female",FinalMarks>80)
student %>%filter(Gender=="Female"|FinalMarks>90)
student %>%filter(Department %in% c("Statistics","Economics"))

student %>% arrange(Department, desc(FinalMarks))
student %>% filter(FinalMarks>=80) %>% select(Name, Department, FinalMarks)
student %>% filter(Department=="Statistics") %>% arrange(desc(FinalMarks)) %>% select(Name,FinalMarks)
##################Missing##
student1 <- data.frame(
  StudentID=c(101,102,103,104,105,106),
  Name=c("Rahim","Karim","Jannat","Nadia", "Sakib","Mim"),
  Age=c(20,21,NA,22,20,NA),
  Marks=c(85,NA,90,88,75,92),
  Gender=c("Male","Male","Female","Female","Male","Female")
)
student1
is.na(student1)
is.na(student1$Marks)
sum(is.na(student1))
colSums(is.na(student1))
rowSums(is.na(student1))

complete.cases(student1)
student1[!complete.cases(student1),]
student_clean <- na.omit(student1)
student_clean <-student1[complete.cases(student1),]





##Lec-7
#remove all objects from the current R environment (workspace).
rm(list = ls())
student <- data.frame(
  StudentID=c(101,102,103,104,105,106,107,108),
  Name=c("Rahim","Karim","Jannat","Nadia",
         "Sakib","Mim","Arif","Suma"),
  Gender=c("Male","Male","Female","Female",
           "Male","Female","Male","Female"),
  Department=c("Statistics","Statistics",
               "Economics","Computer Science",
               "Statistics","Economics",
               "Mathematics","Statistics"),
  Age=c(20,21,19,22,20,21,23,20),
  Marks=c(85,92,78,88,65,95,75,81)
)
head(student)
names(student)
dim(student)
duplicated(student)
student[duplicated(student),]
student <- unique(student)
library(dplyr)
student <- student %>%
  rename(FinalMarks=Marks)
names(student)
student <- student %>%
  mutate(Result=ifelse(FinalMarks>=40,"Pass", "Fail"))
head(student)
student <- student %>%
  mutate(Grade=case_when(FinalMarks>=80~"A", FinalMarks>=70~"B",FinalMarks>=60~"C", TRUE~"F"))
head(student)
student$Name
student[,2]
student[,c(2,4)]
student[1:3,]
student %>%filter(FinalMarks>80)
student %>%filter(Gender=="Female")
student %>%filter(Department=="Statistics")
student %>%filter(Gender=="Female",FinalMarks>80)
student %>%filter(Gender=="Female"|FinalMarks>90)
student %>%filter(Department %in% c("Statistics","Economics"))
student %>% arrange(Department, desc(FinalMarks))
student %>% filter(FinalMarks>=80) %>% select(Name, Department, FinalMarks)
student %>% filter(Department=="Statistics") %>% arrange(desc(FinalMarks)) %>% select(Name,FinalMarks)

##################Missing##
student1 <- data.frame(
  StudentID=c(101,102,103,104,105,106),
  Name=c("Rahim","Karim","Jannat","Nadia", "Sakib","Mim"),
  Age=c(20,21,NA,22,20,NA),
  Marks=c(85,NA,90,88,75,92),
  Gender=c("Male","Male","Female","Female","Male","Female")
)
student1
new_student <- data.frame(
  StudentID = 107,
  Name = "Nadia",
  Gender = "Female",
  Age = 22,
  Marks = 88
)
data=rbind(student1,new_student)
data
new_students <- data.frame(
  StudentID = c(108,109,110),
  Name = c("Nadia","Sakib","Mim"),
  Gender = c("Female","Male","Female"),
  Age = c(22,20,21),
  Marks = c(88,65,95)
)

student2 <- rbind(student1, new_students)
student2

student3=rbind(student1,new_student,new_students)
student3

is.na(student1)
is.na(student1$Marks)
sum(is.na(student1))
colSums(is.na(student1))
rowSums(is.na(student1))

student1[!complete.cases(student1),]
student_clean <- na.omit(student1)
student_clean <-student1[complete.cases(student1),]

student$Age[is.na(student$Age)] <-mean(student$Age, na.rm=TRUE)
d=mean(student1$Age, na.rm=TRUE)
d
##selects only the missing Age values:student1$Age[is.na(student1$Age)] 
student1$Age[is.na(student1$Age)] <-d
student1$Age
###na.rm = remove NA values
d1=median(student1$Marks, na.rm=TRUE)
d1
student1$Marks[is.na(student1$Marks)] <-d1
student1$Marks
##########Count the Frequency of Each Category
table(student1$Gender)

##########3
install.packages("mice")
library(mice)

imp <- mice(student1, m = 5)
completed_data <- complete(imp)

################
#Add a new column named Department.(simple):
student1$Department <- c("Statistics",
                         "Economics","Statistics","Statistics",
                         "Mathematics",
                         "Computer Science")
student1
###Using cbind()
Department <- c("Statistics",
                "Economics","Statistics","Statistics",
                "Mathematics",
                "Computer Science")
student1=cbind(student1,Department)
###########Add more column:
student1 <- data.frame(
  StudentID=c(101,102,103,104,105,106),
  Name=c("Rahim","Karim","Jannat","Nadia", "Sakib","Mim"),
  Age=c(20,21,NA,22,20,NA),
  Marks=c(85,NA,90,88,75,92),
  Gender=c("Male","Male","Female","Female","Male","Female")
)
student1

library(dplyr)
student1 <- student1 %>%
  mutate(
    Total = Marks + 10,
    Result = ifelse(Marks >= 40, "Pass", "Fail"),
    Grade = case_when(
      Marks >= 80 ~ "A",
      Marks >= 70 ~ "B",
      Marks >= 60 ~ "C",
      TRUE ~ "F"
    )
  )
student1





###Practice
student <- data.frame(
  StudentID = c(101,102,103,104,105,106,107,108,109,110),
  Name = c("Rahim","Karim","Jannat","Nadia","Sakib","Mim","Arif","Suma","Rafi","Lima"),
  Gender = c("Male","Male","Female","Female","Male","Female","Male","Female","Male","Female"),
  Department = c("Statistics","Statistics","Economics","Computer Science",
                 "Statistics","Economics","Mathematics","Statistics",
                 "Economics","Computer Science"),
  Age = c(20,21,NA,22,20,21,23,20,22,NA),
  Marks = c(85,92,78,88,65,95,75,81,NA,89)
)
# 1. Display the dataset
student

# 2. Show the first six observations
head(student)

# 3. Display the structure
str(student)

# 4. Count rows and columns
nrow(student)      # Number of rows
ncol(student)      # Number of columns
dim(student)       # Rows and Columns together

# 5. Rename Marks to FinalMarks
student<-student%>%rename(FinalMarks=Marks)

# 6. Create Result and Grade variables
student<-student%>%mutate(
  Result=ifelse(FinalMarks>40,"Pass","Fail"),
  Grade=case_when(FinalMarks>=80~"A",FinalMarks>=70~"B",TRUE~"F")
)

student

# 7. Students scoring at least 80 marks
student<-student%>%filter(FinalMarks>80)

# 8. Display female students
student<-student%>%filter(student$Gender=="Female")

# 9. Statistics students scoring above 80
student<-student%>%filter(student$Department=="Statistics"&student$FinalMarks>80)
##OR
student[student$Department == "Statistics" &
          student$FinalMarks > 80, ]

# 10. Sort students by marks (descending)
student%>%arrange(desc(student$FinalMarks))

# 11. Count missing values by column
colSums(is.na(student))

# 12. Replace missing ages with mean age
student$Age[is.na(student$Age)]<-mean(student$Age,na.rm=TRUE)

mean_age <- mean(student$Age, na.rm = TRUE)

student$Age[is.na(student$Age)] <- mean_age

# 13. Replace missing marks with median marks
student$FinalMarks[is.na(student$FinalMarks)]<-median(student$FinalMarks,na.rm=TRUE)
##OR

median_marks <- median(student$FinalMarks, na.rm = TRUE)
student$FinalMarks[is.na(student$FinalMarks)] <- median_marks

# 14. Remove duplicate records
student <- unique(student)

# 15. Save the cleaned dataset
write.csv(student,
          "C://Users//Asus//Downloads//practice.csv",
          row.names = FALSE)

# View the cleaned dataset
View(student)


##Assignment
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
write.csv(students, "C://Users//Asus//Downloads//students.csv",
          row.names=FALSE)


