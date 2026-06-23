CREATE DATABASE if not exists cus;
USE cus;
CREATE TABLE department (
	dept_id INT PRIMARY KEY,
	dept_name VARCHAR(50)
);
CREATE TABLE student(
    student_id INT PRIMARY KEY,
    name VARCHAR(50),
    age INT,
    dept_id INT,
	foreign key(dept_id) references department(dept_id)
);
insert into department values(10,'stat');
insert into department values(20,'math');
alter table student 
add email varchar(100);
insert into student values(1,'rafi','10','20','rafi11@gmail.com');
insert into student values(2,'shafi','20','10','shafi12@gmail.com');
SELECT* FROM student;