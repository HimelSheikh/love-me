CREATE DATABASE IF NOT EXISTS university_db;
USE university_db;

CREATE TABLE students (
    student_id INT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    age INT,
    department VARCHAR(30),
    gpa FLOAT
);

CREATE TABLE teachers (
    teacher_id INT PRIMARY KEY,
    name VARCHAR(50),
    department VARCHAR(30)
);

ALTER TABLE students
ADD email VARCHAR(50);

INSERT INTO students(student_id, name, age, department, gpa, email)
VALUES
(101, 'Rahim', 20, 'CSE', 3.80, 'rahim@gmail.com'),
(102, 'Karim', 21, 'CSE', 3.20, 'karim@gmail.com'),
(103, 'Nila', 22, 'EEE', 2.90, 'nila@gmail.com'),
(104, 'Mim', 20, 'BBA', 3.50, 'mim@gmail.com'),
(105, 'Sakib', 23, 'CSE', 3.90, 'sakib@gmail.com');

INSERT INTO teachers(teacher_id, name, department)
VALUES
(1, 'Dr. Hasan', 'CSE'),
(2, 'Dr. Alam', 'EEE'),
(3, 'Dr. Khan', 'BBA');

SELECT * FROM students;


SELECT name, department, gpa
FROM students
WHERE gpa > 3.00;

SELECT AVG(gpa) AS Average_GPA
FROM students;

SELECT MAX(gpa) AS Maximum_GPA
FROM students;

SELECT MIN(gpa) AS Minimum_GPA
FROM students;

SELECT department, COUNT(*) AS Total_Students
FROM students
GROUP BY department;

SELECT department, COUNT(*) AS Total_Students
FROM students
GROUP BY department
HAVING COUNT(*) > 2;

SELECT s.student_id,
       s.name AS Student_Name,
       s.department,
       t.name AS Teacher_Name
FROM students s
INNER JOIN teachers t ON s.department = t.department;


DELIMITER //

CREATE PROCEDURE InsertStudent(
    IN p_student_id INT,
    IN p_name VARCHAR(50),
    IN p_age INT,
    IN p_department VARCHAR(30),
    IN p_gpa FLOAT,
    IN p_email VARCHAR(50)
)
BEGIN
    INSERT INTO students
    VALUES (
        p_student_id,
        p_name,
        p_age,
        p_department,
        p_gpa,
        p_email
    );
END //

DELIMITER ;

-- Example call
CALL InsertStudent(
    106,
    'Rafi',
    21,
    'EEE',
    3.60,
    'rafi@gmail.com'
);

