from typing import List
from pydantic import BaseModel, conlist

class Item(BaseModel):
    raw: str
    sentiment: List[conlist(float)]