CREATE DATABASE GRADES;
USE GRADES;
CREATE TABLE students_xml (
    sid INT PRIMARY KEY,
    data TEXT
);

CREATE TABLE exercises_xml (
    id INT AUTO_INCREMENT PRIMARY KEY,
    data TEXT
);

CREATE TABLE results_xml (
    id INT AUTO_INCREMENT PRIMARY KEY,
    data TEXT
);
INSERT INTO students_xml VALUES
(101, '<student><first_name>Ann</first_name><last_name>Smith</last_name><email>ann@gmail.com</email></student>'),
(102, '<student><first_name>David</first_name><last_name>Jones</last_name><email>jones@gmail.com</email></student>'),
(103, '<student><first_name>Paul</first_name><last_name>Miller</last_name><email>paul@gmail.com</email></student>'),
(104, '<student><first_name>Maria</first_name><last_name>Brown</last_name><email>maria@gmail.com</email></student>');

INSERT INTO exercises_xml (data) VALUES
('<exercise><cat>HW</cat><eno>1</eno><topic>ER</topic><max_point>10</max_point></exercise>'),
('<exercise><cat>HW</cat><eno>2</eno><topic>SQL</topic><max_point>10</max_point></exercise>'),
('<exercise><cat>Mid</cat><eno>1</eno><topic>SQL</topic><max_point>14</max_point></exercise>');

INSERT INTO results_xml (data) VALUES
('<result><sid>101</sid><cat>HW</cat><eno>1</eno><points>10</points></result>'),
('<result><sid>101</sid><cat>HW</cat><eno>2</eno><points>8</points></result>'),
('<result><sid>101</sid><cat>Mid</cat><eno>1</eno><points>12</points></result>'),
('<result><sid>102</sid><cat>HW</cat><eno>1</eno><points>9</points></result>'),
('<result><sid>102</sid><cat>HW</cat><eno>2</eno><points>9</points></result>'),
('<result><sid>102</sid><cat>Mid</cat><eno>1</eno><points>10</points></result>'),
('<result><sid>103</sid><cat>HW</cat><eno>1</eno><points>5</points></result>'),
('<result><sid>103</sid><cat>Mid</cat><eno>1</eno><points>7</points></result>');

UPDATE students_xml
SET data = '<student><first_name>David</first_name><last_name>Jones</last_name><email>david@gmail.com</email></student>'
WHERE sid = 102;

SELECT
    ExtractValue(data, '/student/first_name') AS first_name,
    ExtractValue(data, '/student/last_name') AS last_name
FROM students_xml;

SELECT
    ExtractValue(data, '/result/points') AS points
FROM results_xml
WHERE ExtractValue(data, '/result/cat') = 'Mid';
