# Healthcare Insurance Analysis
/* 
1. To gain a comprehensive understanding of the factors influencing hospitalization costs, it is necessary to 
   combine the tables provided. Merge the two tables by first identifying the columns in the data tables that will 
   help you in merging.
   a. In both tables, add a Primary Key constraint for these columns
   { Hint: You can remove duplicates and null values from the column and then use ALTER TABLE to add a Primary Key
         constraint. }  */
         
create database insurance ;
use insurance;
create table Hospitalization(
      Customer_ID varchar(10) primary key,
      year int,
      month varchar(5),
      date int,
      children int,
      charges float,
      hospital_tier varchar(10),
      city_tier varchar(10),
      state_id varchar(5)
);
show tables;
select * from insurance.hospitalization;

create table medical(
      Customer_ID varchar(10) primary key,
      BMI float,
      hba1c float,
      heart_issues varchar(5),
      any_transplants varchar(5),
      cancer_history varchar(5),
      number_of_surgeries varchar(24),
      smoker varchar(5)
);
select * from insurance.medical;

SELECT * FROM insurance.hospitalization INNER JOIN insurance.medical 
ON hospitalization.Customer_ID = medical.customer_id;

select (2023-year) as Age from insurance.hospitalization; 

/*
2. Retrieve information about people who are diabetic and have heart problems with their average age, the average
   number of dependent children, average BMI, and average hospitalization costs.    */
SELECT AVG(2023-h.year) AS avg_age, AVG(h.children) AS avg_num_children, AVG(h.charges) AS avg_hospitalization_cost,
	   AVG(m.bmi) AS avg_bmi
FROM insurance.hospitalization h inner join insurance.medical m on h.customer_id=m.customer_id
WHERE m.hba1c > 6.5  AND m.heart_issues = "yes";

SELECT (2023-h.year) AS age, (h.children) AS num_children, (h.charges) AS hospitalization_cost,
	   (m.bmi) AS bmi
FROM insurance.hospitalization h inner join insurance.medical m on h.customer_id=m.customer_id
WHERE m.hba1c > 6.5  AND m.heart_issues = "yes";

# 3. Find the average hospitalization cost for each hospital tier and each city level. 
SELECT AVG(charges) AS AverageHospitalizationCost, Hospital_Tier, City_Tier 
FROM insurance.hospitalization 
GROUP BY Hospital_Tier, City_Tier
order by City_Tier;
      
# 4. Determine the number of people who have had major surgery with a history of cancer.
SELECT COUNT(*) 
FROM insurance.medical 
WHERE cancer_history = "yes" AND number_of_surgeries > 0;  

# 5. Determine the number of tier-1 hospitals in each state.
SELECT state_id, COUNT(*) AS num_tier_1_hospitals 
FROM insurance.hospitalization 
WHERE hospital_tier = "tier - 1" 
GROUP BY state_id 
ORDER BY num_tier_1_hospitals DESC;
