# Creating a very first thing that is state

#1 Using typed Dictionary
from typing import TypedDict
class State(TypedDict):
    topic : str
    summary : str
    score : str


#2 Using Pydantic Model
from pydantic import BaseModel, field_validator
class state(BaseModel):
    topic : str
    score : int
    summary : str = ""

    @field_validator
    def score_positive(cls,v):
        if v < 0:
            raise ValueError("Score must be positive")


#3 python data classes

from dataclasses import dataclass,field

@dataclass
class State:
    topic : str = ""
    summary : str = ""
    messages : list = field(default_factory = list)


# 4 Using langgraph default
from langgraph.graph import MessagesState
class State(MessagesState):
    user_name: str
    language: str