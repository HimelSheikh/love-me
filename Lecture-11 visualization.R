x <- c(1, 2, 3, 4,5,8,9)
plot(x)
dev.off()
##
par(mfrow = c(1, 2))
plot(1:5, main = "Plot 1",col="green",pch=19)
plot(5:1, main = "Plot 2",col="red",pch=18)
plot(1:10, main = "Plot 3",col="red",pch=17)
plot(10:1, main = "Plot 4",col="red",pch=15)
#dev.off()
x <- c(1, 2, 3, 4, 5)
y <- c(2, 5, 4, 8, 10)

plot(x, y,
     main = "Scatter Plot",
     xlab = "X values",
     ylab = "Y values",col="skyblue",
     pch = 19)
#########
x <- 1:5
y <- c(2, 5, 4, 8, 10)

plot(x, y,
     pch = 19,
     col = "blue",
     main = "Scatter Plot",
     xlab = "X",
     ylab = "Y")
##
df <- data.frame(
  height = c(150, 160, 165, 170, 180),
  weight = c(50, 58, 62, 70, 80)
)
plot(df$height, df$weight,
     main = "Height vs Weight",  xlab = "Height",
     ylab = "Weight",col = "Red",  pch = 19)
###########
x <- 1:5
y <- c(2, 5, 4, 8, 10)

plot(x, y,
     pch = 19,
     col = c("red", "blue", "green", "orange", "purple"),
     main = "Scatter Plot")
###########3
x <- 1:10
y <- c(3, 5, 4, 7, 8, 10, 9, 12, 11, 14)

group <- c("A", "A", "A", "A", "A",
           "B", "B", "B", "B", "B")

plot(x, y,
     pch = 19,
     col = ifelse(group == "A", "blue", "red"),
     main = "Scatter Plot by Group",
     xlab = "X",
     ylab = "Y")

legend("topleft",
       legend = c("Group A", "Group B"),
       col = c("blue", "red"),
       pch = 19)
#############
library(ggplot2)
ggplot(df, aes(x = height, y = weight)) +
  geom_point() +
  labs(
    title = "Height vs Weight",
    x = "Height",
    y = "Weight"
  ) +
  theme_minimal()
###
# Create some sample data
x <- 1:10
y <- x^2
plot(x, y)

# Add a vertical line at x = 5
abline(v = 5)

# Add a horizontal line at y = 25
abline(h = 25)

# Add a diagonal line with slope 1 and intercept 0
abline(a = 0, b = 1)

# Add a line using the lines() function
x2 <- 1:10
y2 <- 2*x2 + 3
lines(x2, y2, col = "red", lty = 2)
##
#Basic line graph
x <- 1:6
y <- c(10, 15, 12, 20, 18, 25)

plot(x, y,
     type = "l",
     main = "Line Graph",
     xlab = "Time",
     ylab = "Value")
###
month <- c("Jan", "Feb", "Mar", "Apr", "May", "Jun")
rainfall <- c(12, 18, 35, 80, 220, 310)

plot(rainfall,
     type = "o",
     pch = 19,
     col = "blue",
     #xaxt = "n",  #removes default x-axis
     main = "Monthly Rainfall",
     xlab = "Month",
     ylab = "Rainfall (mm)")

axis(1, at = 1:6, labels = month)
###Add 2nd lline
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
     main = "Monthly Rainfall Comparison")

lines(month, rainfall_2026,
      type = "o",
      pch = 17,
      col = "red")

legend("topleft",
       legend = c("2025", "2026"),
       col = c("blue", "red"),
       pch = c(19, 17))
##
v <- c(17, 25, 38, 13, 41)
t <- c(22, 19, 36, 19, 23)
m <- c(25, 14, 16, 34, 29)

plot(v, type = "l", col = "red", xlab = "Month", 
     ylab = "Articles Written", 
     main = "Articles Written Chart")
lines(t, type = "o", col = "blue")
lines(m, type = "s", col = "green")
##simple Bar chart
A <- c(17, 32, 8, 53, 1)
barplot(A, xlab = "X-axis", ylab = "Y-axis",col="red", main ="Bar-Chart")
##
students <- c(40, 55, 35, 60)
departments <- c("Statistics", "Botany", "Pharmacy", "Chemistry")
barplot(students,
        names.arg = departments,
        main = "Number of Students by Department",
        xlab = "Department",
        ylab = "Number of Students",col="Red")
##
barplot(students,
        names.arg = departments,
        col = c("blue", "red", "green", "orange"),
        main = "Number of Students by Department",
        xlab = "Department",
        ylab = "Number of Students")
##
sex <- c("Male", "Female", "Female", "Male",
         "Female", "Female", "Male", "Female")

freq <- table(sex)
barplot(freq,
        col = c("skyblue", "pink"),
        main = "Students by Sex",
        xlab = "Sex",
        ylab = "Frequency")
##Add Values on Top of Bars
students <- c(40, 55, 35, 60)
bp <- barplot(students,
              names.arg = departments,
              col = "skyblue",
              main = "Number of Students",
              ylab = "Students",
              ylim = c(0, 70))

text(bp, students,
     labels = students,
     pos = 3)
##
A <- c(17, 2, 8, 13, 1, 22)
B <- c("Jan", "Feb", "Mar", "Apr", "May", "Jun")

barplot(A, names.arg = B, xlab = "Month",
        ylab = "Articles", col = "steelblue",
        main = "GeeksforGeeks - Article Chart",
        cex.main = 1.5, cex.lab = 1.2, cex.axis = 1.1)

text(
  x = barplot(A, names.arg = B, col = "steelblue", ylim = c(0, max(A) * 1.2)),
  y = A + 1, 
  labels = A, 
  pos = 3, 
  cex = 1.2, 
  col = "black"
)
##############
# Data
students <- matrix(c(25, 30,
                     15, 20,
                     20, 25,
                     15, 12),
                   nrow = 2,
                   byrow = TRUE)
students
# Department names
colnames(students) <- c("Statistics",
                        "Botany",
                        "Pharmacy",
                        "Chemistry")

# Gender names
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
                   las = 3)
# Add values above bars
text(x = bar,
     y = students,
     labels = students,
     pos = 3,
     cex = 0.5)


# Add legend
legend("topright",
       legend = c("Male", "Female"),
       fill = c("skyblue", "pink"),
       border = "black",
       bty = "o",
       cex = 0.5)
dev.off()
