CREATE DATABASE department_db;
USE department_db;

CREATE TABLE SGroups (
    group_id INT PRIMARY KEY,
    name VARCHAR(50)
);

-- Students Table
CREATE TABLE Students (
    student_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    group_id INT,
    FOREIGN KEY (group_id) REFERENCES SGroups(group_id)
);

-- Subjects Table
CREATE TABLE Subjects (
    subject_id INT PRIMARY KEY,
    title VARCHAR(100)
);

-- Teachers Table
CREATE TABLE Teachers (
    teacher_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50)
);

-- Relationship Table
CREATE TABLE Subject_Teacher (
    subject_id INT,
    teacher_id INT,
    group_id INT,
    PRIMARY KEY (subject_id, teacher_id, group_id),
    FOREIGN KEY (subject_id) REFERENCES Subjects(subject_id),
    FOREIGN KEY (teacher_id) REFERENCES Teachers(teacher_id),
    FOREIGN KEY (group_id) REFERENCES Groups(group_id)
);

-- Marks Table
CREATE TABLE Marks (
    mark_id INT PRIMARY KEY,
    student_id INT,
    subject_id INT,
    date DATETIME,
    mark INT,
    FOREIGN KEY (student_id) REFERENCES Students(student_id),
    FOREIGN KEY (subject_id) REFERENCES Subjects(subject_id)
);