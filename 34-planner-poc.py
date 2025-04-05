from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

system="""
You are a resourceful planner.

You will be given an [objective], and your job is to
prepare an executive plan to complete the [objective].

You should follow the steps below to prepare your plan
1. evaluate the [objective], determine whether it is fesible to do it or not, 
if it cannot be done, just ask for revise the objective.
2. If you find the [objective] or any is unclear, you can do a research on that.
3. for each step in the plan, please provide the instruction and the expected outcome.
4. you will be given [resoures] to help execute the plan
5. you don't need to care the timeline of the plan

finally the plan should be present in the markdown format
 
"""
user="""
[objective]
Given an employee profile, find 3-5 openings which are best fit to the
employee

[resources]
You are given the following resources to help you to complete the goal
- Your profile which includes your background, skills, working experience, education, and qualification
- A database which have all job openings which found from the internet
"""

model=ChatOllama(
    model="llama3.2",
    temperature=0
)

template=ChatPromptTemplate([
    ('system',system),
    ('user',user),
])

chain=template|model

response=chain.invoke({})

print(response)