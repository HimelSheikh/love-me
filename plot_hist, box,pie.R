v <- c(19, 23, 11, 5, 16, 21, 32, 14, 19, 27, 39)

hist(v, xlab = "No.of Articles", col = "green",
     border = "black", xlim = c(0, 50),
     ylim = c(0, 5), breaks = 5)
# Generate random data
set.seed(123)
x <- rnorm(1000, mean = 50, sd = 10)
# Create histogram
hist(x,
     main = "Histogram of Data",
     xlab = "Value",
     ylab = "Frequency")
####
hist(x,
     breaks = 20,
     main = "Histogram of Normally Distributed Data",
     xlab = "Value",
     ylab = "Frequency",
     col = "lightblue",
     border = "black")
#Histogram with density curve
hist(x,
     breaks = 20,
     probability = TRUE,
     main = "Histogram with Normal Density Curve",
     xlab = "Value",
     col = "lightblue",
     border = "black")

curve(dnorm(x, mean = 50, sd = 10),
      add = TRUE,
      lwd = 3)
##skewed
set.seed(123)
x <- rexp(1000, rate = 1)
hist(x,
     breaks = 30,
     probability = TRUE,
     main = "Right-Skewed Distribution",
     xlab = "X",
     ylab = "Density",
     col = "lightblue",
     border = "black")
###
set.seed(123)
x <- -rexp(1000, rate = 1)
hist(x,
     breaks = 30,
     probability = TRUE,
     main = "Left-Skewed Distribution",
     xlab = "X",
     ylab = "Density",
     col = "lightblue",
     border = "black")

##compare all
par(mfrow = c(1, 3))

# Right-skewed
x <- rexp(1000)
hist(x, breaks = 30,
     main = "Right-Skewed",
     xlab = "X")

# Symmetric
x <- rnorm(1000)
hist(x, breaks = 30,
     main = "Symmetric",
     xlab = "X")

# Left-skewed
x <- -rexp(1000)
hist(x, breaks = 30,
     main = "Left-Skewed",
     xlab = "X")
par(mfrow = c(1, 1))
##
dev.off()
par(mfrow=c(2,2))
x <- rnorm(1000, mean = 50, sd = 10)
hist(x, breaks = 10)
hist(x, breaks = 5)
hist(x, breaks = 20)
hist(x, breaks = 50)
##
a=data("airquality")
str(airquality)
Temperature <- airquality$Temp
hist(Temperature,main="Maximum daily temperature at La Guardia Airport",
     xlab="Temperature in degrees Fahrenheit",
     xlim=c(50,100),
     col="darkmagenta",
     freq=FALSE)
# create a histogram of the "Temperature" variable
h <- hist(Temperature,col="chocolate",
          border="brown")
# print the histogram object
print(h)
text(h$mids,h$counts,labels=h$counts, adj=c(0.5, -0.5))
#####
students <- c(40, 30, 20, 10)
departments <- c("Statistics", "Botany",
                 "Pharmacy", "Chemistry")
pie(students,
    labels = departments,
    main = "Students by Department")
pie(students,
    labels = labels,
    col = c("skyblue", "lightgreen",
            "orange", "pink"),
    main = "Students by Department")
students <- c(40, 30, 20, 10)
departments <- c("Statistics", "Botany",
                 "Pharmacy", "Chemistry")

percentage <- round(students / sum(students) * 100, 1)

labels <- paste(departments, percentage, "%")

pie(students,
    labels = labels,
    main = "Students by Department")
# create a named vector 'expenditure'
expenditure <- c(Housing = 600, Food = 300, Cloths = 150, Entertainment = 100, Other = 200)
# print the 'expenditure' vector
print(expenditure)
pie(expenditure)
#Pie chart with additional parameters
pie(expenditure,
    labels=as.character(expenditure),
    main="Monthly Expenditure Breakdown",
    col=c("red","orange","yellow","blue","green"),
    border="brown",
    clockwise=TRUE
)
#####box-plot
marks <- c(45, 50, 52, 55, 58, 60, 62, 65, 68, 70, 72, 75)
boxplot(marks,
        main = "Box Plot of Marks",
        ylab = "Marks")
##outlier
marks <- c(45, 50, 52, 55, 58, 60, 62, 65,
           68, 70, 72, 75, 120)
boxplot(marks,
        main = "Box Plot with Outlier",
        ylab = "Marks",
        col = "lightblue")
###different groups
statistics <- c(65, 70, 72, 75, 78, 80, 82, 85)
botany <- c(55, 60, 62, 65, 68, 70, 72, 75)
pharmacy <- c(70, 72, 75, 78, 80, 82, 85, 88)

marks <- list(Statistics = statistics,
              Botany = botany,
              Pharmacy = pharmacy)
boxplot(marks,
        main = "Marks by Department",
        xlab = "Department",
        ylab = "Marks",
        col = c("lightblue", "lightgreen", "orange"))
###horizontal
boxplot(marks,
        horizontal = TRUE,
        main = "Marks by Department",
        xlab = "Marks")
##show five summary
boxplot.stats(marks$Statistics)
###
boxplot(airquality$Ozone)
#We can see that data above the median is more dispersed. We can also notice two outliers at the higher extreme.
boxplot(airquality$Ozone,
        main = "Mean ozone in parts per billion at Roosevelt Island",
        xlab = "Parts Per Billion",
        ylab = "Ozone",
        col = "orange",
        border = "brown",
        horizontal = TRUE,
        notch = TRUE
)#The narrow part around the median is the notch.
# prepare the data
ozone <- airquality$Ozone
temp <- airquality$Temp
# gererate normal distribution with same mean and sd
ozone_norm <- rnorm(200,mean=mean(ozone, na.rm=TRUE), sd=sd(ozone, na.rm=TRUE))
temp_norm <- rnorm(200,mean=mean(temp, na.rm=TRUE), sd=sd(temp, na.rm=TRUE))
#Now we make 4 boxplots with this data. 
#We use the arguments at and names to denote the place and label.
boxplot(ozone, ozone_norm, temp, temp_norm,
        main = "Multiple boxplots for comparision",
        at = c(1,2,4,5),
        names = c("ozone", "normal", "temp", "normal"),
        ##
boxplot(Temp~Month,
                data=airquality,
                main="Different boxplots for each month",
                xlab="Month Number",
                ylab="Degree Fahrenheit",
                col="orange",
                border="brown"),        
        las = 2,
        col = c("orange","red"),
        border = "brown",
        horizontal = TRUE,
        notch = TRUE
)
marks <- c(45, 50, 52, 55, 58, 60, 62, 65, 68, 70, 72, 75)
stripchart(marks,
           main = "Strip Chart of Marks",
           xlab = "Marks")
stripchart(marks,
           method = "jitter",
           main = "Strip Chart of Marks",
           xlab = "Marks")
stripchart(marks,
           method = "jitter",
           vertical = TRUE,
           main = "Strip Chart of Marks",
           ylab = "Marks")
male <- c(55, 60, 62, 65, 68, 70, 72)
female <- c(58, 62, 65, 67, 70, 73, 75)

stripchart(list(Male = male, Female = female),
           method = "jitter",
           vertical = TRUE,
           main = "Marks by Gender",
           ylab = "Marks")
stripchart(marks, method = "jitter")
stripchart(marks, method = "stack")
##
# prepare the data
temp <- airquality$Temp
# gererate normal distribution with same mean and sd
tempNorm <- rnorm(200,mean=mean(temp, na.rm=TRUE), sd = sd(temp, na.rm=TRUE))
# make a list
x <- list("temp"=temp, "norm"=tempNorm)
stripchart(x,
           main="Multiple stripchart for comparision",
           xlab="Degree Fahrenheit",
           ylab="Temperature",
           method="jitter",
           col=c("orange","red"),
           pch=16
)
stripchart(Temp~Month,
           data=airquality,
           main="Different strip chart for each month",
           xlab="Months",
           ylab="Temperature",
           col="brown3",
           group.names=c("May","June","July","August","September"),
           vertical=TRUE,
           pch=16
)
#colors
# create a vector 'temp' with numeric values
temp <- c(5, 7, 6, 4, 8)
# Plot 1: Bar plot with default settings
barplot(temp, main = "By default")
# Plot 2: Bar plot with custom coloring
barplot(temp, col = "coral", main = "With coloring")
barplot(temp, col=rainbow(5), main="rainbow")
barplot(temp, col=heat.colors(5), main="heat.colors")
barplot(temp, col=terrain.colors(5), main="terrain.colors")
barplot(temp, col=topo.colors(5), main="topo.colors")
##########R plot function
x <- seq(-pi,pi,0.1)
plot(x, sin(x),main="The Sine Function",
                   ylab="sin(x)", col="red",type="l")
plot(x, sin(x),
     main="Overlaying Graphs",
     ylab="",
     type="l",
     col="blue")
lines(x,cos(x), col="red")
legend("topleft",
       c("sin(x)","cos(x)"),
       fill=c("blue","red"),cex=0.8
)

