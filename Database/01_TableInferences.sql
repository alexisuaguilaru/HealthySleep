CREAte SCHEMA IF NOT EXISTS ML;

CREATE TABLE IF NOT EXISTS ML.ModelInferences (
    id SERIAL PRIMARY KEY,

    Gender VARCHAR(6) NOT NULL 
        CHECK (Gender IN ('Male', 'Female')),

    Age SMALLINT NOT NULL 
        CHECK (Age >= 0),

    Occupation VARCHAR(30) NOT NULL 
        CHECK (Occupation IN ('Software Engineer', 'Doctor', 'Sales Representative', 
                              'Teacher', 'Nurse', 'Engineer', 'Accountant', 'Scientist', 
                              'Lawyer', 'Salesperson', 'Manager')),

    SleepDuration NUMERIC(3, 2) NOT NULL 
        CHECK (SleepDuration >= 0 AND SleepDuration <= 24),

    PhysicalActivityLevel SMALLINT NOT NULL 
        CHECK (PhysicalActivityLevel >= 0 AND PhysicalActivityLevel <= 100),

    StressLevel SMALLINT NOT NULL 
        CHECK (StressLevel >= 1 AND StressLevel <= 10),

    BMICategory VARCHAR(10) NOT NULL
        CHECK (BMICategory IN ('Normal', 'Overweight', 'Obese')),
        
    HeartRate SMALLINT NOT NULL 
        CHECK (HeartRate >= 0 AND HeartRate <= 220),

    DailySteps INTEGER NOT NULL 
        CHECK (DailySteps >= 0),

    SleepDisorder VARCHAR(15) NOT NULL
        CHECK (SleepDisorder IN ('No', 'Sleep Apnea', 'Insomnia')),
        
    BloodPressureSystolic SMALLINT NOT NULL 
        CHECK (BloodPressureSystolic >= 0),

    BloodPressureDiastolic SMALLINT NOT NULL
        CHECK (BloodPressureDiastolic >= 0)
);