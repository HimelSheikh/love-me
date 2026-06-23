CREATE DATABASE university_er;
USE university_er;
CREATE TABLE student (
    student_id INT PRIMARY KEY,
    name VARCHAR(50),
    email VARCHAR(50) UNIQUE,
    department VARCHAR(30)
);
select * from student;
CREATE TABLE instructor (
    instructor_id INT PRIMARY KEY,
    instructor_name VARCHAR(50),
    office VARCHAR(20)
);
CREATE TABLE course (
    course_id INT PRIMARY KEY,
    course_name VARCHAR(50),
    credits INT,
    instructor_id INT,
    FOREIGN KEY (instructor_id)
        REFERENCES instructor(instructor_id)
);
CREATE TABLE enrollment (
    enrollment_id INT PRIMARY KEY,
    student_id INT,
    course_id INT,
    semester VARCHAR(20),
    grade CHAR(2),
    FOREIGN KEY (student_id)
        REFERENCES student(student_id),
    FOREIGN KEY (course_id)
        REFERENCES course(course_id)
);
INSERT INTO instructor VALUES
(1, 'Dr. Rahman', 'Room 301');

INSERT INTO course VALUES
(101, 'Database Systems', 3, 1);

INSERT INTO student VALUES
(201, 'Amina', 'amina@gmail.com', 'CSE');

INSERT INTO enrollment VALUES
(1, 201, 101, 'Spring 2025', 'A');
SELECT s.name, c.course_name
FROM student s
JOIN enrollment e ON s.student_id = e.student_id
JOIN course c ON e.course_id = c.course_id;
SELECT i.instructor_name, c.course_name
FROM instructor i
JOIN course c ON i.instructor_id = c.instructor_id;
SELECT s.name, c.course_name, e.grade
FROM enrollment e
JOIN student s ON e.student_id = s.student_id
JOIN course c ON e.course_id = c.course_id;
SELECT * FROM student_result;