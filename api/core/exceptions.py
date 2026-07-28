class APIException(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code

class ModelLoadError(APIException):
    def __init__(self, message="Machine learning model failed to load."):
        super().__init__(message, status_code=500)

class DatabaseError(APIException):
    def __init__(self, message="Database operation failed."):
        super().__init__(message, status_code=500)
