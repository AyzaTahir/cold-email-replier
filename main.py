from crewai import Agent,Task,Crew,LLM
from dotenv import load_dotenv
import os

load_dotenv()

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key = os.getenv("GROQ_API_KEY")
)

game_designer_agent=Agent(
    role="game designer developer",
    goal="design a test base game for the user to play using python that will use pygame library tp create a game",
    backstory="You are an experienced game designer with expertise in creating engaging game mechanics and storylines.",
    llm=llm 
)

game_designer_task=Task(
    description="you are a game designer developer which have 15 years of experience in gaming industry and love to build a text based games for kids and teens.",
    expected_output="you are game designer developer",
    agent=game_designer_agent
)   

crew = Crew(
    agents=[game_designer_agent],
    tasks=[game_designer_task],
    llm=llm
)       

results = crew.kickoff()
print(results)