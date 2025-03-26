import operator
import json
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

@tool
def getJobDescription(id: str) -> dict[str,any]:
    """Get Job Description by ID"""
    with open('data/jd_'+id+'.json','r') as file:
        data = json.load(file)
    return data 

@tool
def getCandidateProfile(id: str) -> dict[str,any]:
    """Get Candidate Profile by ID"""
    with open('data/cand_'+id+'.json','r') as file:
        data = json.load(file)
    return data

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

class AgentState(TypedDict):
    jobDescId: str
    candidateId: str
    characteristics: list[str] 
    msg: Annotated[list[AnyMessage],operator.add]

class Agent:
    def __init__(self):
        graph = StateGraph(AgentState)
        graph.add_node("template",self.promptTemplateNode)
        graph.add_node("model",self.modelNode)
        graph.add_node("tools",self.toolsNode)

        graph.add_edge(START,"template")
        graph.add_edge("template","model")
        graph.add_conditional_edges(
            "model",
            self.needsToolsCalling,
            {True:"tools",False:END}) # map return from needsToolsCalling to next node
        
        graph.add_edge("tools","model")
        self.graph = graph.compile()

        tools = [getJobDescription,getCandidateProfile]
        self.toolsDict = {
            "getJobDescription": getJobDescription,
            "getCandidateProfile": getCandidateProfile
        }

        ollamaModel = "llama3.2"

        self.model = ChatOllama(
            model=ollamaModel,
            temperature=0.1).bind_tools(tools)

    def promptTemplateNode(self, state:AgentState) -> AgentState:
        template = ChatPromptTemplate([
            ("system",
            f"""
            Your role: {role}
            Your task: {task}
            Instruction for execute your task: {instruction}
            """),
            ("user","Job Description ID: {jd_id}, Candidate ID: {cand_id}") #, Charactertistics: {characteristics}")
        ])
        return {'msg': template.format_messages(
            jd_id=state['jobDescId'],
            cand_id=state['candidateId']#,
            #characteristics=state['characteristics']
        )}

    def modelNode(self, state:AgentState) -> AgentState:
        print('here')
        prevMsgs = state['msg']
        aiMsg = self.model.invoke(prevMsgs)
        return {'msg': [aiMsg]}

    def toolsNode(self, state:AgentState) -> AgentState:
        print('here2')
        lastAIMsg = state['msg'][-1]
        toolMsgs = []
        for t in lastAIMsg.tool_calls:
            print(t)
            tool = self.toolsDict[t['name']]
            toolMsg = tool.invoke(t)
            print(toolMsg)
            toolMsgs.append(toolMsg)
        return {'msg': toolMsgs}

    def needsToolsCalling(self, state:AgentState) -> bool:
        lastMsg = state['msg'][-1]
        return len(lastMsg.tool_calls) > 0
    
    def invoke(self, jobDescId:str, candidateId:str, characteristics:list[str]=[]):
        return self.graph.invoke({
            'jobDescId': jobDescId,
            'candidateId': candidateId,
            'characteristics': characteristics
        })

agent = Agent()
result = agent.invoke("A","A")
print(result)