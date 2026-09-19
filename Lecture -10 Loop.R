age <- c(20, 25, 30, 35, 40, 45)
gender <- c(
  "Male", "Female", "Male",
  "Female", "Male", "Female"
)
#Calculate mean age by gender:
  tapply(age, gender, max)
  age <- c(20, 25, 30, 35, 40, 45)
  sex <- c(
    "Male", "Female", "Male",
    "Female", "Male", "Female"
  )
  region <- c(
    "Urban", "Urban", "Rural",
    "Rural", "Urban", "Rural"
  )
  tapply(age, list(sex, region), mean)
  data <- data.frame(
    age = c(20, 25, 30, 35),
    income = c(20000, 25000, 30000, 40000),
    score = c(70, 80, 75, 90))
  #Calculate the mean of every numeric column:
    lapply(data, mean) 
    x <- list(
      a = 1:3,
      b = 4:6,
      c = 7:9
    )
    lapply(x, function(z) z^2)
###LOOPING
    for (i in 1:5) {
      print(i)
    }
    #Print even numbers:
      for (i in 1:20) {
        if (i %% 2 == 0) {
          print(i)
        }
      }
    #Print odd numbers:
      for (i in 1:20) {
        if (i %% 2 != 0) {
          print(i)
        }
      }
    students <- c("Rahim", "Karim", "Sumaiya", "Nusrat")
    for (name in students) {
      print(name)
    }
   # For Loop with a Character Vector:
      subjects <- c("Statistics", "R", "Python", "Database")
    for (subject in subjects) {
      print(subject)
    }
#for loop with a sequence      
for (x in seq(2, 10, 2)) {
        print(x)
      }
#For Loop for Calculations
for (i in 1:5) {square <- i^2
        print(square)
      }
#Calculate Sum Using a For Loop
x <- c(10, 20, 30, 40, 50)
total <- 0
for (i in x) {
total <- total + i
      }
print(total)      
print(i) 
##
x <- c(10, 20, 30, 40, 50)
for (i in 1:length(x)) {
  print(x[i])
}
#For Loop and Conditional Statements
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
#Nested for
for (i in 1:5) {
  for (j in 1:10) {
    print(paste(i, "×", j, "=", i * j))
  }
}
marks <- c(55, 67, 72, 81, 90)
#We want to identify whether each student has passed.
for (mark in marks) {
  if (mark >= 40) {
    print("Pass")
  } else {
    print("Fail")
  }
}
x <- c(12, 15, 18, 21, 24, 27)
count <- 0
for (i in x) { 
  if (i %% 2 == 0) {
    count <- count + 1
  }
}
print(count)
#########
i <- 1
while (i <= 5) {
  print(i)
  i <- i + 1
}
i <- 1
while (i <= 5) {
  print(i)
  i <- i + 1
}
#####
i <- 1
total <- 0
while (total <= 100) {
  total <- total + i
  i <- i + 1
}
print(i - 1)
print(total)
#####
i <- 1
repeat {
  print(i)
  i <- i + 1
  if (i > 5) {
    break
  }
  print(i)
}


