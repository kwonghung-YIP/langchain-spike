import getpass
import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

@tool
def getJobDescription(id: str) -> dict[str,any]:
    """Get Job Description by ID"""
    with open('data/jd_'+id+'.json','r') as file:
        data = json.load(file)
    return data #json.dump(data)

@tool
def getCandidateProfile(id: str) -> dict[str,any]:
    """Get Candidate Profile by ID"""
    with open('data/cand_'+id+'.json','r') as file:
        data = json.load(file)
    return data #json.dump(data)

def run_agent(chain,toolsDict,userInput):
    steps = []
    userInput["agent_steps"] = steps
    i = 0
    while i < 5:
        i = i + 1
        aiMsg = chain.invoke(userInput)

        print('---')
        print(aiMsg)

        if not aiMsg.tool_calls:
            return aiMsg
        else:
            steps.append(aiMsg)
            for t in aiMsg.tool_calls:
                #print('---')
                tool = toolsDict[t['name']]
                toolMsg = tool.invoke(t)
                #print(toolMsg)
                steps.append(toolMsg)

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

role = "You are a experienced developer seeking from Hong Kong seeking for job in UK IT industry"
task = """
You found an opening and you need to prepare a profile summary for your resume to apply the job.
The summary should highlight your skills which matched with requirements from the Job Description.
"""
instruction = """
If characteristics are provided, please including it in the summary.
Try to be creative and make the summary eye catching.
Both Job Description and your Candidate Profile are from external resource, they are in JSON format
The result should include the following and shoud in JSON format:
summary: your profile summary less than 500 words
skills: an array of skill matches in the Job Description
"""

template = ChatPromptTemplate([
    ("system",
     """
     Your role: {role}
     Your task: {task}
     Instruction for execute your task: {instruction}
     """),
    ("user","Job Description ID: {jd_id}, Candidate ID: {cand_id}, Charactertistics: {characteristics}"),
    ("placeholder","{agent_steps}")
])

tools = [getJobDescription,getCandidateProfile]
toolsDict = {
    "getJobDescription": getJobDescription,
    "getCandidateProfile": getCandidateProfile
}

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5).bind_tools(tools)

chain = template | model 

response = run_agent(chain,toolsDict,{
    "role":role,"task":task,"instruction":instruction,
    "jd_id":"A","cand_id":"A",
    "characteristics":["strong technical background", "resourceful", "fast learner"]
})

print(response)

print('---')
print(response.content)
print(response.tool_calls)

