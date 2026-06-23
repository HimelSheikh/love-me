create database recur;
use recur;
CREATE TABLE prereq (
    course_id  VARCHAR(8),
    prereq_id  VARCHAR(8),
    PRIMARY KEY (course_id, prereq_id)
);
INSERT INTO prereq VALUES
('CS-190', 'CS-101'),
('CS-315', 'CS-190'),
('CS-319', 'CS-315'),
('CS-347', 'CS-319');
select*from prereq;
WITH RECURSIVE rec_prereq(course_id, prereq_id) AS (
    -- Direct prerequisites
    SELECT course_id, prereq_id
    FROM prereq

    UNION

    -- Indirect prerequisites
    SELECT r.course_id, p.prereq_id
    FROM rec_prereq r
    JOIN prereq p
      ON r.prereq_id = p.course_id
)
SELECT prereq_id
FROM rec_prereq
WHERE course_id = 'CS-347';

use recur;
-#---Create course Table (needed for foreign keys)
CREATE TABLE course (
    course_id VARCHAR(8) PRIMARY KEY,
    course_name VARCHAR(50)
);

-- Sample courses
INSERT INTO course VALUES
('BIO-101', 'Intro to Biology'),
('BIO-301', 'Advanced Biology'),
('BIO-399', 'Special Biology'),
('CS-101', 'Intro to CS'),
('CS-190', 'Programming Fundamentals'),
('CS-315', 'Data Structures'),
('CS-319', 'Algorithms'),
('CS-347', 'Advanced CS');
select* from course;
#
drop table prereq;
CREATE TABLE prereq1 (
    course_id VARCHAR(8),
    prereq_id VARCHAR(8),
    PRIMARY KEY (course_id, prereq_id),
    FOREIGN KEY (course_id) REFERENCES course(course_id)
        ON DELETE CASCADE,
    FOREIGN KEY (prereq_id) REFERENCES course(course_id)
);
INSERT INTO prereq1 VALUES ('BIO-301', 'BIO-101');
INSERT INTO prereq1 VALUES ('BIO-399', 'BIO-101');
INSERT INTO prereq1 VALUES ('CS-190', 'CS-101');
INSERT INTO prereq1 VALUES ('CS-315', 'CS-190');
INSERT INTO prereq1 VALUES ('CS-319', 'CS-101');
INSERT INTO prereq1 VALUES ('CS-319', 'CS-315');
INSERT INTO prereq1 VALUES ('CS-347', 'CS-319');
SELECT * FROM prereq1;
WITH RECURSIVE rec_prereq(course_id, prereq_id) AS (
    -- Anchor: direct prerequisites
    SELECT course_id, prereq_id
    FROM prereq1

    UNION

    -- Recursive: prerequisites of prerequisites
    SELECT r.course_id, p.prereq_id
    FROM rec_prereq r
    JOIN prereq1 p
      ON r.prereq_id = p.course_id
)
SELECT prereq_id
FROM rec_prereq
WHERE course_id = 'CS-347';