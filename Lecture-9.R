gender <- factor(c("Male", "Female", "Female", "Male", "Male"))
gender
gender1 <- c("Male", "Female", "Female", "Male", "Male")
gender1
class(gender1)
is.factor(gender)
class(gender)
levels(gender)
nlevels(gender)
education <- factor(
  c("Primary", "Secondary", "Graduate", "Graduate", "Primary")
)
education
risk <- factor(
  c("Low", "High", "Medium", "Low"),
  levels = c("Low", "Medium", "High"),
  ordered = TRUE
)
risk
risk[1] < risk[2]
student <- data.frame(
  ID = 1:6,
  Name = c("Rahim", "Karim", "Nila", "Sadia", "Hasan", "Rafi"),
  Gender = c("Male", "Male", "Female", "Female", "Male", "Male"),
  Department = c("Statistics", "Mathematics",
                 "Statistics", "Physics",
                 "Mathematics", "Statistics")
)
student
student$Gender <- factor(student$Gender)
student$Department <- factor(student$Department)
str(student)
student$Gender <- factor(
  student$Gender,
  levels = c("Female", "Male")
)
table(student$Gender, student$Department)
gender <- factor(
  gender,
  levels = c("F", "M"),
  labels = c("Female", "Male")
)
gender <- factor(
  c("M", "F", "F", "M"),
  levels = c("F", "M"),
  labels = c("Female", "Male")
)
gender
x <- factor(
  c("1", "2", "3"),
  levels = c("1", "2", "3"),
  labels = c("Low", "Medium", "High")
)
x
