create database if not exists person;
use person;
CREATE TABLE Persons (
    ID int NOT NULL,
    LastName varchar(255) NOT NULL,
    FirstName varchar(255),
    Age int,
    PRIMARY KEY (ID)
);
CREATE TABLE Persons1 (
    ID int ,
    LastName varchar(255) ,
    FirstName varchar(255),
    Age int,
    CONSTRAINT PK_Person PRIMARY KEY (ID,LastName)
);
select* from persons1;
CREATE TABLE Orders (
    OrderID int NOT NULL,
    OrderNumber int NOT NULL,
    ID int,
    PRIMARY KEY (OrderID),
    FOREIGN KEY (ID) REFERENCES Persons(ID)
);
select* from Orders;
### New table
CREATE TABLE Persons2 (
    PersonID int,
    LastName varchar(255),
    FirstName varchar(255),
    Address varchar(255),
    City varchar(255)
);
select* from Persons2;
### creates a new table called "Test" (which is a copy of the "Persons2" table)
CREATE TABLE Test AS
SELECT PersonID, City
FROM Persons2;
select* from Test;