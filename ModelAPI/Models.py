from re import findall
from pydantic import BaseModel , Field , ConfigDict , AliasGenerator
from typing import Literal

RegexFieldNameRepresentation = r'[A-Z][a-z]+'
class InputMLModel(BaseModel):
    Gender: Literal['Male', 'Female'] = Field(...,)

    Age: int = Field(...,ge=0)

    Occupation: Literal['Software Engineer', 'Doctor', 'Sales Representative', 
                        'Teacher', 'Nurse', 'Engineer', 'Accountant', 'Scientist', 
                        'Lawyer', 'Salesperson', 'Manager'] = Field(...,)

    SleepDuration: float = Field(...,ge=0,le=24)

    PhysicalActivityLevel: int = Field(...,ge=0,le=100)

    StressLevel: int = Field(...,ge=1,le=10)

    BMICategory: Literal['Normal','Overweight','Obese'] = Field(...,serialization_alias='BMI Category')

    HeartRate: int = Field(...,ge=0,le=300)

    DailySteps: int = Field(...,ge=0)

    SleepDisorder: Literal['No','Sleep Apnea','Insomnia'] = Field(...,)

    BloodPressureSystolic: int = Field(...,ge=0)

    BloodPressureDiastolic: int = Field(...,ge=0)

    model_config = ConfigDict(
        alias_generator = AliasGenerator(
            serialization_alias = lambda field_name : ' '.join(findall(RegexFieldNameRepresentation,field_name))
        )
    )

class OutputMLModel(BaseModel):
    LevelQualitySleep: int = Field(...,ge=0)

    NameQualitySleep: str = Field(...)