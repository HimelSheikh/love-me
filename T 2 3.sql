CREATE TABLE Person (
    PersonID int PRIMARY KEY,
    LastName varchar(255),
    FirstName varchar(255),
    Age int
);
INSERT INTO Person (PersonID, LastName, FirstName, Age)
VALUES 
(1, 'Hansen', 'Ola', 30),
(2, 'Svendson', 'Tove', 23),
(3, 'Pettersen', 'Kari', 20);
select*from person;

CREATE TABLE Orders1 (
    OrderID int PRIMARY KEY,
    OrderNumber int NOT NULL,
    PersonID int,
    FOREIGN KEY (PersonID) REFERENCES Person(PersonID)
);
INSERT INTO Orders1 (OrderID, OrderNumber, PersonID)
VALUES 
(1, 77895, 3),
(2, 44678, 3),
(3, 22456, 2),
(4, 24562, 1);
select*from orders1;
