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
