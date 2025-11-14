from re import findall
from pydantic import BaseModel , Field , ConfigDict , AliasGenerator , computed_field
from typing import Literal

RegexFieldNameRepresentation = r'[A-Z][a-z]+'
class InputMLModel(BaseModel):
    """
    Data model for defining the input features of a patient.
    """

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

NamesLevelQualitySleep = {
    4: 'Deficient',
    5: 'Poor',
    6: 'Acceptable',
    7: 'Good',
    8: 'Optimal',
    9: 'Regenerative',
}
class OutputMLModel(BaseModel):
    """
    Data model for defining the predicted output for a patient.
    """

    LevelQualitySleep: int = Field(...,ge=4,le=9)

    @computed_field
    @property
    def NameQualitySleep(self) -> str:
        return NamesLevelQualitySleep[self.LevelQualitySleep]