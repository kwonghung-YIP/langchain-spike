import json
from ollama import chat, embed
from ollama import ChatResponse
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup

class RoleRequirement(BaseModel):
    education: list[str] = Field(description="Minimum education requirement")
    experience: list[str] = Field(description="Working experience requirement")
    technicalSkill: list[str] = Field(description="Technical skills requirement")
    softSkill: list[str] = Field(description="Soft skills requirement")
    niceToHave: list[str] = Field(description="Any skills that counts as bonus")

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
    'type':'function',
    'function': {
        'name': 'get_job_description',
        'description': 'Get the job description by jobId',
        'parameters': {
            'type': 'object',
            'properties': {
                'jobId': {
                    'type':'str',
                    'description':'jobId for searching the job description record'
                },
            },
            'required': ['jobId'],
        }
    }
}

class ToolAgent:
    def __init__(self):
        self._messages = [
            { 'role':'system','content':toolAgentSystemInstruction},
        ]
        self.toolsDef = [get_job_description]
        self.toolsMap = {
            'get_job_description': getJobDescription
        }
    
    def invoke(self, jobId: str):
        self._messages.append({'role':'user','content':f'jobId={jobId}'})

        haveToolCall: bool = True
        while haveToolCall:
            response: ChatResponse=chat(
                model='llama3.2',
                messages=self._messages,
                tools=self.toolsDef,
                #format=JobDescription.model_json_schema(),
                options={
                    'temperature':0.0
                }
            )

            #print(response['message']['content'])
            print(response['message'])

            haveToolCall=False
            message=response['message']
            if message.tool_calls:
                for toolcall in message.tool_calls:
                    if tool := self.toolsMap.get(toolcall.function.name):
                        #print(toolcall.function.name)
                        #print(toolcall.function.arguments)
                        output = tool(**toolcall.function.arguments)
                        #print(output)
                        self._messages.append({'role':'tool','content':str(output),'name':toolcall.function.name})
                        haveToolCall=True
                    else:
                        print(f"function not found:{toolcall.function.name}")
        
        return response

class JsonOutputAgent:    
    def format(self,textResult:str) -> ChatResponse:
        response: ChatResponse=chat(
            model='llama3.2',
            messages=[
                {'role':'system','content':outputAgentSystemInstruction},
                {'role':'user','content':textResult},
            ],
            format=JobDescription.model_json_schema(),
            options={
                'temperature':0.2
            }
        )
        #print(response)
        #self._messages.append(response['message'])
        return response

def flow(jobId: str):
    toolAgent = ToolAgent()
    resp1 = toolAgent.invoke(jobId)
    #print(resp1)

    outputAgent = JsonOutputAgent()
    resp2 = outputAgent.format(resp1['message']['content'])

    content = resp2['message']['content']
    print(content)

    #resp3 = embed(model='llama3.2',input=content)
    #print(resp3['embeddings'])

#flow('54722144')
flow('54722077')
flow('54722158')