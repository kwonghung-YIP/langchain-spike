import json
import operator
from typing import TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from pydantic import BaseModel, Field

system="""
You are an agent try to solve a given problem.
You will be assigned a <role> and a <task>, and you have to follow the given <plan> 
to complete your task. 
A list of resources helping you to complete your task are included in <resources>.
When you execute the plan, you must follow the limitation defined in <constraints>. 
The expected outcome id defined in <expectation>, the result should be presented in
the given format.
If you found that you cannot complete the task for any reason, please stop and tells
that you cannot complete the task.
All parameters above are within corresponding XML tag. 
"""

user="""
<role>
You are a job agent specifise on UK IT market
</role> 

<task>
- Your task is to find up to 3 job openings from the provided sources which match 
with the given candidate profile, and you also need to consider the expectation
from the candidate.
- If more than one job openings have been found, you should give a ranking and 
present the result showing the best fit first.
- For each result, you have to provide a profile summary to show why the candidate
fits to the job, and the summary must be less than 500 words. 
</task>

<constraints>
- You should only search the job opening from the provided vector database. If you 
cannot find any openings, or you cannot reach the vector database, do not make up any 
openings and just report that no job opening is available.
</constraints>

<plan>
- Analyze the candidate profile, identify his/her working experience, soft and
technical skills, education background, and professional qualification, and present 
your result in JSON format.
- Understand the expectation from the candidate, search job openings from the provided
sources. You can refine your query or question to get more precise result.
- Validate the candidate profile against the search result, ensure that the candidate 
profile could meet with the minimum requirements.
- If multiple job openings were found in above steps, give a ranking to each result, and
return up to 3 result. If no job openings were found, just stop and report that no available 
opening for that candidate, DO NOT make up and job opening which is not from the provided
sources.
- Prepare a profile summary for each job opening, emphasize how the experience, skills and
background of the candidate fit with the job opening.
</plan>

<expectation>
- Max up to 3 job openings, and quote their source (jobId).
- profile summary for each openings, less than 500 words.
</expectation>

<resources>
sources of job openings
- by calling the tool "searchJobOpening"
</resources>
"""

prompt="""
The candidate profile Id is {candidateId}
"""

@tool
def getCandidateProfile(id: str) -> dict[str,any]:
    """Get Candidate Profile by ID"""
    with open('data/cand_'+id+'.json','r') as file:
        data = json.load(file)
    return data #json.dump(data)

class SearchJobOpeningInput(BaseModel):
    question: str = Field(description="question for job openings search")

@tool(args_schema=SearchJobOpeningInput)
def searchJobOpening(
    question: str
) -> list[Document]:
    """
    This function support vector embedding and semantic search for job opening,
    it returns a list of document if any job openings in the vector database matches
    to your question, otherwise it return an empty list which means no job openings
    are available matches with your question.
    """
    print(f"query for searching job openings:{question}")
    return []


class AgentState(TypedDict):
    candidateId: str
    msg: Annotated[list[AnyMessage],operator.add]

class Agent:
    def __init__(self):
        graph = StateGraph(AgentState)
        graph.add_node("prompt",self.promptNode)
        graph.add_node("model",self.modelNode)
        graph.add_node("tools",self.toolsNode)
    
        graph.add_edge(START,"prompt")
        graph.add_edge("prompt","model")
        graph.add_conditional_edges(
            "model",
            self.route,
            {
                "call_tools":"tools",
                "done": END,
                "others":"model"
            }
        )
        graph.add_edge("tools","model")
        self.graph = graph.compile()

        self.tools={
            "getCandidateProfile":getCandidateProfile,
            "searchJobOpening":searchJobOpening
        }

        self.model=ChatOllama(
            model="llama3.2",
            temperature=0
        ).bind_tools(self.tools.values())

    def promptNode(self, state:AgentState) -> AgentState:
        template=ChatPromptTemplate([
            ("system",system),
            ("system",user),
            ("user",prompt)
        ])
        return {
            'msg': template.format_messages(
                candidateId=state['candidateId']
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
            try:
                # print(t)
                tool = self.tools[t['name']]
                toolMsg = tool.invoke(t)
                # print(toolMsg)
                toolMsgs.append(toolMsg)
            except:
                print(lastAIMsg)
                raise
        return {'msg': toolMsgs}
    
    def route(self, state:AgentState) -> str:
        lastMsg=state['msg'][-1]
        if len(lastMsg.tool_calls) > 0:
            return "call_tools"
        elif lastMsg.response_metadata['done']:
            return "done"
        else:
            return "others"

    def invoke(self, candidateId:str):
        return self.graph.invoke({
            'candidateId': candidateId
        })

agent=Agent()
state=agent.invoke("A")

print(searchJobOpening.args_schema)

for m in state['msg']:
    print(m)