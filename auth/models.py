from pydantic import BaseModel

class SignupRequest(BaseModel):
    username:str
    password:str
    role:str