def SplitBloodPressure(
        BloodPressure: str,
    ) -> list[int,int]:
    """
    Function for splitting blood pressure 
    measure into systolic and diastolic 
    pressure.

    Parameter
    ---------
    BloodPressure: str
        Blood pressure of a patient in 
        Systolic/Diastolic form

    Returns
    -------
    Systolic: int
        Systolic pressure measure

    Diastolic: int
        Diastolic pressure measure
    """

    return list(map(int,BloodPressure.split('/')))