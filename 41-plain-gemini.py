import os, getpass, json
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Reference: https://ai.google.dev/gemini-api/docs/text-generation

def getJobDescription(jobId: str) -> dict[str,any]:
    """Get Job Description by jobId"""
    #print(f"getJobDescription:{jobId}")
    with open(f"reed.co.uk/job-details/job_{jobId}.json",'r') as file:
        data = json.load(file)

    if data['salary']:
        salary=BeautifulSoup(data['salary'],features="lxml")
        data['salary']=salary.get_text()

    jobDesc=BeautifulSoup(data['jobDescription'],features="lxml")
    data['jobDescription']=jobDesc.get_text()

    return data

get_job_description = {
    'name': 'get_job_description',
    'description': 'Get the job description by jobId',
    'parameters': {
        'type': 'object',
        'properties': {
            'jobId': {
                'type':'string',
                'description':'jobId for searching the job description record'
            },
        },
        'required': ['jobId'],
    }
}

class JobDescription(BaseModel):
    jobTitle: str = Field(description="Job Title")
    jobType: str = Field(description="Job Type, e.g. full-time, part-time")
    location: str = Field(description="Location")
    roleDescription: list[str] = Field(description="Role duties, about this role")
    requirements: list[str] = Field(description="Requirements for this role")
    package: list[str] = Field(description="Offered salary range and benefit, leave this property empty ")
    tags: list[str] = Field(description="Tags or Keywords of the job description")
    original: str = Field(description="The original job description")
    additionalInfo: str = Field(description="Any additional information discovered by you")
    suggestion: str = Field(description="Any suggestion to make this job description more completed")

toolAgentSystemInstruction = f"""
***Role***
You are a job agent who reviews client provided job description
and form into a standard format

***Task***
- analyze the given job description
- extract the key information
- append a summary for the job description
- include the original job description in the response
- suggest tags, labels for the job description 
"""

outputAgentSystemInstruction = f"""
format the text input into given JSON Schema
"""

class ToolAgent:
    def __init__(self):
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        toolsDef = types.Tool(function_declarations=[
            get_job_description
        ])
        self._toolsMap = {
            'get_job_description': getJobDescription
        }
        self._modelConfig = types.GenerateContentConfig(
            system_instruction=toolAgentSystemInstruction,
            tools=[toolsDef]
        )

    def invoke(self, jobId:str):
        contents = [
            types.Content(role="user",parts=[types.Part(text=f"jobId:{jobId}")])
        ]
        haveToolCall: bool = True
        while haveToolCall:
            response = self._client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=self._modelConfig
            )

            haveToolCall = False
            #print(response.candidates[0].content.parts[0])
            func_call = response.candidates[0].content.parts[0].function_call
            if func_call:
                if tool := self._toolsMap.get(func_call.name):
                    #print(func_call.name)
                    #print(func_call.args)
                    output = tool(**func_call.args)
                    #print(output)

                    contents.append(types.Content(
                        role="model",
                        parts=[types.Part(function_call=func_call)]
                    ))
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part.from_function_response(
                            name=func_call.name,
                            response={
                                "result":output
                            }
                        )]
                    ))

                    haveToolCall=True
                else:
                    print(f"function not found:{func_call.name}")
        
        return response

class JsonOutputAgent:
    def __init__(self):
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def format(self,textResult:str):
        contents=[
            textResult
        ]
        response = self._client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config = {
                'system_instruction': outputAgentSystemInstruction,
                'response_mime_type': 'application/json',
                'response_schema': JobDescription,
            }
        )
        return response

def flow(jobId: str):
    toolAgent = ToolAgent()
    response = toolAgent.invoke(jobId)
    #print(response)
    print(response.candidates[0].content.parts[0])
    outputAgent = JsonOutputAgent()
    response2 = outputAgent.format(response.candidates[0].content.parts[0].text)
    print(response2)
    return response2.candidates[0].content.parts[0].text
if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")

result = flow('54722077')
print(f"""
****
{result}
****
""")