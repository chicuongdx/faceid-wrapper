from typing import Union
from pydantic import BaseModel

class FaceId(BaseModel):
    def __init__(self, faceid, distance):
        self.faceid = faceid
