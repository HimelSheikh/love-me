#COMPLEX DATA TYPES (PART I) – Semi-structured Data, XML, JSON
#XML Data Handling fully in MySQL, with table creation, data insertion, queries, and updates.
#XML (eXtensible Markup Language) is a text-based hierarchical format.
create database employee;
use employee;
CREATE TABLE employees_xml (
    emp_id INT PRIMARY KEY,
    emp_info TEXT
);

INSERT INTO employees_xml VALUES 
(1, '<employee><name>Rahim</name><age>30</age><dept>IT</dept></employee>');

SELECT ExtractValue(emp_info, '/employee/name') AS Name,
       ExtractValue(emp_info, '/employee/age') AS Age
FROM employees_xml;
#################################
CREATE TABLE staff_xml (
    staff_id INT PRIMARY KEY,
    emp_info TEXT
);
drop table staff_xml;
INSERT INTO staff_xml VALUES
(1, '<employee><name>Rahim</name><age>30</age><dept>IT</dept><skills><skill>SQL</skill><skill>Python</skill></skills></employee>'),
(2, '<employee><name>Karim</name><age>28</age><dept>CSE</dept><skills><skill>HTML</skill><skill>CSS</skill></skills></employee>'),
(3, '<employee><name>Sadia</name><age>25</age><dept>IT</dept><skills><skill>Python</skill><skill>Java</skill></skills></employee>');
SELECT ExtractValue(emp_info, '/employee/name') AS Name,
       ExtractValue(emp_info, '/employee/dept') AS Department
FROM staff_xml;
#Change Karim’s department from CSE → HR
UPDATE staff_xml
SET emp_info = UpdateXML(emp_info, '/employee/dept', 'HR')
WHERE staff_id = 2;
SELECT ExtractValue(emp_info, '/employee/skills/skill[1]') AS FirstSkill
FROM staff_xml
WHERE staff_id = 1;
#Extract all skills
SELECT ExtractValue(emp_info, '/employee/skills/skill[1]') AS Skill1,
       ExtractValue(emp_info, '/employee/skills/skill[2]') AS Skill2
FROM staff_xml
WHERE staff_id = 1;
##################
<student>
    <student_id>101</student_id>  
    <name>Sadia</name>
    <age>22</age>
    <department>CSE</department>
</student>
################################
#Store and Query JSON
                      #NEXT CLASS
# JSON data
CREATE TABLE students_json (
    student_id INT PRIMARY KEY,
    info JSON
);
INSERT INTO students_json VALUES
(1, '{"name": "Rahim", "age": 20, "dept": "IT", "skills": ["SQL","Python"]}'),
(2, '{"name": "Karim", "age": 22, "dept": "CSE", "skills": ["HTML","CSS"]}'),
(3, '{"name": "Sadia", "age": 21, "dept": "IT", "skills": ["Python","Java"]}');
SELECT info->>'$.name' AS Name,
       info->>'$.dept' AS Department
FROM students_json;
SELECT info->>'$.name' AS Name
FROM students_json
WHERE info->>'$.dept' = 'IT';
#Update Rahim’s age to 21
UPDATE students_json
SET info = JSON_SET(info, '$.age', 21)
WHERE student_id = 1;
select *from students_json;
SELECT JSON_EXTRACT(info, '$.age') AS age
FROM students_json;
SELECT 
    JSON_EXTRACT(info, '$.age') AS age,
    JSON_EXTRACT(info, '$.name') AS name,
    JSON_EXTRACT(info, '$.dept') AS dept
FROM students_json;
SELECT 
    info->>'$.age' AS age,
    info->>'$.name' AS name,
    info->>'$.dept' AS dept
FROM students_json;
                                                        #Next Class
 #Spatial Data Handling
 drop table hospital_location1;
CREATE TABLE hospital_location1 (
    hospital_id INT PRIMARY KEY,
    name VARCHAR(50),
    location POINT NOT NULL,
    SPATIAL INDEX(location) 
);

INSERT INTO hospital_location1 VALUES
(1, 'City Hospital', POINT(23.8103, 80.4125)),
(2, 'Central Hospital', POINT(23.8150, 80.4200)),
(3, 'North Hospital', POINT(23.8250, 80.4100));
SELECT hospital_id,
       name,
       ST_X(location) AS latitude,
       ST_Y(location) AS longitude
FROM hospital_location1;
SELECT name,
       ST_Distance_Sphere(location, POINT(23.8120, 80.4150)) AS distance_meters
FROM hospital_location1
WHERE ST_Distance_Sphere(location, POINT(23.8120, 80.4150)) <= 1000;
SELECT name,
       ST_Distance_Sphere(location, POINT(23.8120, 80.4150)) AS distance_meters
FROM hospital_location1
ORDER BY distance_meters ASC
LIMIT 1;




SELECT name,
       ST_Distance_Sphere(location, POINT(23.8120, 80.4150)) AS Distance_meters
FROM hospital_location1
WHERE ST_Distance_Sphere(location, POINT(23.8120, 80.4150)) <= 1000;
SELECT name,
       ST_Distance_Sphere(location, POINT(23.8120, 80.4150)) AS Distance_meters
FROM hospital_location1
ORDER BY Distance_meters ASC;
