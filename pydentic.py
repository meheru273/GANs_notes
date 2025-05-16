from pydantic import BaseModel,EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'meheru' # default value
    age: Optional[int] = None # optional field
    grade: float = Field(..., gt=0, le=4) # grade must be between 0 and 4
    email: EmailStr  # email validation
    
new_student = Student(name="John Doe", age=20, grade=3.5,email="abc123@gmail.com")

student = Student(**new_student)