Create database if not exists lecture4;
use lecture4;

CREATE TABLE student53 (
    student_id INT PRIMARY KEY,
    name VARCHAR(50),
    age INT,
    department VARCHAR(50),
    income DECIMAL(10,2),
    cgpa DECIMAL(3,2),
    city VARCHAR(50)
);
INSERT INTO student53 (student_id, name, age, department, income, cgpa, city)
VALUES
(1, 'Amina Rahman', 20, 'Pharmacy', 15000.00, 3.80, 'Dhaka'),

(2, 'Rahim Uddin', 22, 'Statistics', 12000.50, 3.20, 'Chittagong'),

(3, 'Niloy Hasan', 21, 'CSE', 18000.00, 3.90, 'Khulna'),

(4, 'Mita Akter', 23, 'Pharmacy', 10000.00, 2.75, 'Rajshahi'),

(5, 'Sadia Hossain', 19, 'Biology', 9000.00, 3.50, 'Sylhet'),

(6, 'Touhid Alam', 24, 'Mathematics', 20000.00, 3.95, 'Barishal'),

(7, 'Farhana Jahan', 22, 'Statistics', 13000.00, 3.10, 'Comilla'),

(8, 'Sabbir Ahmed', 20, 'CSE', 11000.00, 2.60, 'Rangpur'),

(9, 'Shamima Begum', 21, 'Pharmacy', 14000.00, 3.85, 'Narayanganj'),

(10, 'Ridoy Khan', 23, 'Economics', 16000.00, 3.40, 'Gazipur');
select name, age, cgpa from student53;
select name, age, cgpa from student53 where age>=20;
select name, age, cgpa, department from student53 where department='Pharmacy';
SELECT name, cgpa FROM student53 ORDER BY cgpa DESC;
SELECT name, cgpa FROM student53 ORDER BY cgpa ASC;
SELECT *FROM student53 ORDER BY cgpa ASC;
SELECT COUNT(student_id) FROM student53;
 SELECT AVG(age)FROM student53;
 SELECT SUM(income) FROM student53;
 SELECT MAX(income) AS max_income FROM student53;
 SELECT MIN(income) AS min_income FROM student53;
 SELECT MIN(income) FROM student53;
 SELECT department, COUNT(department) FROM student53 GROUP BY department;
 SELECT age, COUNT(age) FROM student53 GROUP BY age;
 SELECT age, student_id,COUNT(age) FROM student53 GROUP BY student_id;
 SELECT COUNT(income), department, student_id FROM student53 GROUP BY  student_id ORDER BY COUNT(income) DESC;
  SELECT department, COUNT(*) AS total FROM student53 GROUP BY department HAVING COUNT(*) >= 2;
  select *from student53 limit 3;
  select *from student53 limit 3 offset 3;