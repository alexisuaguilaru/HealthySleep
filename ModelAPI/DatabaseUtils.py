from os import getenv
from psycopg2 import connect
from psycopg2.extensions import connection
from .Models import InputMLModel , OutputMLModel

def GetDatabaseConnection() -> connection:
    """
    Function for getting a connection to the database.
    """

    DatabaseConnection = connect(
        dbname = getenv('POSTGRES_DB','sql_database'),
        user = getenv('API_DB_USER','api_user'),
        password = getenv('API_DB_PASSWORD','api_user_pass'),
        host = getenv('DB_HOST','localhost'),
        port = int(getenv('DB_PORT',5432)),
    )

    try:
        yield DatabaseConnection
    except:
        print('FATAL ERROR CONNECTION')
    finally:
        DatabaseConnection.close()

ColumnNames = [*InputMLModel.model_fields.keys(),*OutputMLModel.model_fields.keys()]
ColumnsQuery = ' , '.join(ColumnNames)
ValuePlaceholders = ' , '.join(['%s'] * len(ColumnNames))
InsertQuery = f'INSERT INTO ML.ModelInferences ({ColumnsQuery}) VALUES ({ValuePlaceholders});'
def InsertModelInference(
        InputPatient: InputMLModel,
        ModelInference: OutputMLModel,
        DatabaseConnection: connection,
    ) -> None:
    """
    Function for inserting a patient (features/values and 
    predicted quality of sleep) into the database.

    Parameters
    ----------
    InputPatient: InputMLModel
        Features (Values) of a patient.

    ModelInference: OutputMLModel
        Predicted quality of sleep for a patient.
        
    DatabaseConnection: connection
        Database connection.
    """

    ValuesQuery = [*InputPatient.model_dump().values(),ModelInference.model_dump().get('LevelQualitySleep')]

    SQLCursor = DatabaseConnection.cursor()
    try:
        SQLCursor.execute(InsertQuery,ValuesQuery)
        DatabaseConnection.commit()
    except:
        print('FATAL ERROR INSERT')
        DatabaseConnection.rollback()
    finally:
        SQLCursor.close()