import json
import operator
from typing import TypedDict, Annotated, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser

system="""
<role>
You are a job agent specializing in the UK IT job market. Your purpose is to match
job seekers (candidate) with the most relevant jobs based on their profiles and expectations.
</role>

<action>
Your task is to identify, rank, and summarize up to three jobs that align with a given
candidate's profile and expectations. If no jobs meet the criteria, report that no jobs
are available without fabricating results.
</action>

<instructions>
Follow this structured process:

1. Analyze the candidate's profile: Extract work experience, technical and soft skills, 
education, and qualifications. Output this in JSON format.

2. Understand the candidate’s expectations and refine the job search accordingly.

3. Search for jobs using the provided vector database and ensure they meet the candidate’s 
minimum requirements.

4. Rank up to three jobs by best fit. If no jobs are found, report it without making up any data.

5. Summarize why each selected job fits the candidate, ensuring summaries are concise (≤500 words).

</instructions>

<limits>
- ONLY search for jobs in the provided vector database.

- DO NOT fabricate jobs or modify job details.

- IF searchJobs returns an empty list, respond ONLY with:
  "No job matches found based on the provided candidate profile."
  DO NOT generate alternative job listings or assume missing details.

- STOP immediately and report if no jobs match the candidate’s profile or if the database is unreachable.
</limits>

<SuccessCriteria>
- Maximum of three jobs, ranked by best fit.

- Each job must include a job ID (source).

- Concise summaries (≤500 words) explaining how the candidate is a good fit.

- Report no job is available if no jobs been found in the Vector Database.

</SuccessCriteria>
"""

jsonScheme="""
{
    "title": "ExpectedResult",
    "description": "Job search result",
    "type": "object",
    "properties": {
        "candidate": {
            "description": "Candidate profile",
            "type": "object",
            "properties": {
                "id": {
                    "description": "Candidate Profile ID",
                    "type": "string"
                },
                "name": {
                    "description": "Candidate Full Name",
                    "type": "string"
                }
            },
            "required": ["id","name"]
        },
        "searchResult": {
            "description": "State the search result",
            "type": "string"
        },
        "agentSuggestion": {
            "description": "Suggestion form the job search agent",
            "type": "string"
        }
    }
}
"""

prompt="""
Candidate profile Id: {candidateId}
Candidate expectation: 
{expectations}
"""

@tool
def getCandidateProfile(id: str) -> dict[str,any]:
    """Get Candidate Profile by ID"""
    with open('data/cand_'+id+'.json','r') as file:
        data = json.load(file)
    return data #json.dump(data)

@tool
def searchJobs(
    question: str
) -> list[Document]:
    """
    Searches for job  in the vector database using semantic search.

    Args:
        question (str): A search query describing the desired job role, skills, and requirements.

    Returns:
        list[Document]: A list of job documents matching the query. Each document contains:
            - job_id (str): Unique identifier for the job
            - title (str): Job title
            - company (str): Hiring company name
            - location (str): Job location
            - description (str): Detailed job description
            - requirements (list[str]): List of required skills and qualifications
            - source (str): Where the job listing was found
    
    If no matching jobs are found, an empty list is returned.
    
    Example Usage:
        searchJobs("Python developer with 3 years of experience in London")

    Example Output #1:
        [
            {
                "job_id": "12345",
                "title": "Software Engineer (Python)",
                "company": "TechCorp",
                "location": "London, UK",
                "description": "Looking for a Python developer with 3+ years of experience...",
                "requirements": ["Python", "Django", "AWS"],
                "source": "TechCorp Careers"
            }
        ]

    Example Output #2:
        None #for no matching job
    """
    print(f"query for searching job openings:{question}")
    return None

class Job(BaseModel):
    jobId: str = Field(description="JobId from the vector database")
    ranking: str = Field(description="ranking of the search result by best fit")
    summary: str = Field(description="Profile summary to tell why the candidate fit the job")
    description: str = Field(description="Job Description from vector database")

class Candidate(BaseModel):
    id: str = Field(description="Candidate Id")
    name: str = Field(description="Candidate Name")
    expectations: list[str] = Field(description="Expectations from the candidate")

class SearchResult(BaseModel):
    """Always use this tool to structure your final response to the user"""
    candidate: Candidate = Field(description="Candidate for the job search")
    #jobs: Optional[list[Job]] = Field(description="Job search results, left it empty if no available job")
    result: str = Field(description="Report the search result")

class AgentState(TypedDict):
    candidateId: str
    expectations: list[str]
    msg: Annotated[list[AnyMessage],operator.add]
    result: SearchResult

class Agent:
    def __init__(self):
        graph = StateGraph(AgentState)
        graph.add_node("prompt",self.promptNode)
        graph.add_node("model",self.modelNode)
        graph.add_node("tools",self.toolsNode)
        graph.add_node("output",self.outputNode)
    
        graph.add_edge(START,"prompt")
        graph.add_edge("prompt","model")
        graph.add_conditional_edges(
            "model",
            self.route,
            {
                "call_tools":"tools",
                "done": "output",
                "others":"model"
            }
        )
        graph.add_edge("tools","model")
        graph.add_edge("output",END)
        self.graph = graph.compile()

        self.tools={
            "getCandidateProfile":getCandidateProfile,
            "searchJobs":searchJobs
        }

        self.model=ChatOllama(
            model="llama3.2",
            temperature=0
        ).bind_tools(self.tools.values())

        self.outputModel=ChatOllama(
            model="llama3.2",
            temperature=0
        ).with_structured_output(schema=SearchResult)


    def promptNode(self, state:AgentState) -> AgentState:
        template=ChatPromptTemplate([
            ("system",system),
            ("user",prompt)
        ])
        return {
            'msg': template.format_messages(
                candidateId=state['candidateId'],
                expectations=state['expectations'],
                jsonScheme=jsonScheme
            )
        }
    
    def modelNode(self, state:AgentState) -> AgentState:
        allMsgs = state['msg']
        aiMsg = self.model.invoke(allMsgs)
        return {
            'msg': [aiMsg]
        }
    
    def toolsNode(self, state:AgentState) -> AgentState:
        lastAIMsg = state['msg'][-1]
        toolMsgs = []
        for t in lastAIMsg.tool_calls:
            if t['name'] in self.tools.keys():
                try:
                    # print(t)
                    tool = self.tools[t['name']]
                    toolMsg = tool.invoke(t)
                    # print(toolMsg)
                    toolMsgs.append(toolMsg)
                except:
                    print(lastAIMsg)
                    raise
            else:
                print(f"Cannot find tools {t['name']} from the tools dict")
        return {'msg': toolMsgs}
    
    def outputNode(self, state:AgentState) -> AgentState:
        lastMsg=state['msg'][-1]
        template=ChatPromptTemplate([
        ])
        result=self.outputModel.invoke([
            ('user','format the result in JSON format'),
            lastMsg
        ])
        print(result)
        return {
            'result': result
        }

    def route(self, state:AgentState) -> str:
        lastMsg=state['msg'][-1]
        if len(lastMsg.tool_calls) > 0:
            return "call_tools"
        elif lastMsg.response_metadata['done']:
            return "done"
        else:
            return "others"

    def invoke(self, candidateId:str, expectations:list[str]):
        return self.graph.invoke({
            'candidateId': candidateId,
            'expectations': expectations
        })

agent=Agent()
state=agent.invoke("A",["Working location: London","Job type: full-time","expected salary: ~50K"])

for m in state['msg']:
    print(m)

print(state['result'])