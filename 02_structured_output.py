import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
#llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# --- A1. Pydantic Approach (Full Validation & Fields) ---
class CodeReviewResult(BaseModel):
    is_secure: bool = Field(description="True if the code has no critical vulnerabilities")
    risk_level: str = Field(description="Low, Medium, or High risk assessment")
    suggestions: list[str] = Field(description="List of improvements or fixes")

pydanticA1_llm = llm.with_structured_output(CodeReviewResult)
pydanticA1_output = pydanticA1_llm.invoke("Review this code: eval(user_input)")

print("=== A1. PydanticA1 Output ===")
print(f"Secure: {pydanticA1_output.is_secure}")
print(f"Risk: {pydanticA1_output.risk_level}")
print(f"Suggestions: {pydanticA1_output.suggestions}")

## A2
class Movie(BaseModel):
    name : str = Field(description="Name of the actor or actress")
    role : str = Field(description="Role played by the actor or actress")

class MovieReview(BaseModel):
    name : str = Field(description="Name of the movie")
    cast : list[Movie]
    date_of_release : str = Field(description="Release date of the movie")
    Budget : str = Field(description="Budget of the movie in indian rupees")
    Profit : str = Field(description="profit of the movie in indian rupees")
    director : str = Field(description="director of the movie")
    producer : str = Field(description="producer of the movie")
    genre : str = Field(description="Genre of the movie")

pydanticA2_llm = llm.with_structured_output(MovieReview)
pydanticA2_output = pydanticA2_llm.invoke("Give a review on the malayalam movie lucifer (2019)")

print("=== A2. PydanticA2 Output ===")
print(pydanticA2_output)



# --- B. TypedDict Approach (Lightweight Dictionary Schema) ---
class MovieMetadata(TypedDict):
    title: str
    genre: str
    release_year: int

typedict1_llm_response = llm.with_structured_output(MovieMetadata)
dict1_output = typedict1_llm_response.invoke("Extract info from this text: Christopher Nolan directed the mind-bending sci-fi film Inception back in 2010.")

print("\n=== B1. TypedDict Output ===")
print(dict1_output)

class Elephant(TypedDict):
    name : str
    age : str
    height : str
    owned_by : str

typeddict2_llm_response=llm.with_structured_output(Elephant)
dict2_op=typeddict2_llm_response.invoke("Give me the details of the elephant named Thechikottukavu Ramachandran")
print("\n=== B2. TypedDict Output ===")
print(dict2_op)