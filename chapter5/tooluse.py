#  pip install langchain-classic
import os
import asyncio
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

os.environ["GOOGLE_API_KEY"] = ""

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    print(f"✅ Language model initialized: {llm.model}")
except Exception as e:
    print(f"🛑 Error initializing language model: {e}")
    llm = None

# --- Define a Tool ---
# tool 데코레이터: 이 함수가 AI가 호출할 수 있는 특수한 도구임을 LangChain에 알려줌
@tool
def search_information(query: str) -> str:
    """
    Provides factual information on a given topic. Use this tool to find answers to questions
    like 'What is the capital of France?' or 'What is the weather in London?'.
    """
    print(f"\n--- 🛠️ Tool Called: search_information with query: '{query}' ---")
    # Simulate a search tool with a dictionary of predefined results.
    simulated_results = {
        "weather in london": "The weather in London is currently cloudy with a temperature of 15°C.",
        "capital of france": "The capital of France is Paris.",
        "population of earth": "The estimated population of Earth is around 8 billion people.",
        "tallest mountain": "Mount Everest is the tallest mountain above sea level.",
        "default": f"Simulated search result for '{query}': No specific information found, but the topic seems interesting."
    }
    result = simulated_results.get(query.lower(), simulated_results["default"])
    print(f"--- TOOL RESULT: {result} ---")
    return result

tools = [search_information]


# --- Create a Tool-Calling Agent ---
if llm:
    # This prompt template requires an `agent_scratchpad` placeholder for the agent's internal steps.'
    # agent_scratchpad: AI 모델(LLM)이 문제를 해결할 때, 바로 정답을 내놓는 게 아니라 "생각의 과정"을 적어두는 임시 메모장 같은 공간
    # 즉, 에이전트가 어떤 도구를 쓸지 고민하고, 도구를 실행한 결과값을 받아보고, 다음 행동을 결정하는 모든 중간 단계의 텍스트가 이 {agent_scratchpad}라는 변수에 차곡차곡 쌓이게 됨
    # placeholder: 처음 AI에게 질문을 던질 때는 이 부분이 비어있지만, AI가 "어디 보자, 계산기를 써야겠군" 하고 행동을 시작하면 그 기록이 이 자리에 실시간으로 채워짐
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Create the agent, binding the LLM, tools, and prompt together.
    # create_tool_calling_agent: 모델이 도구를 호출할 수 있는 능력을 갖추도록 연결
    agent = create_tool_calling_agent(llm, tools, agent_prompt)

    # AgentExecutor is the runtime that invokes the agent and executes the chosen tools.
    # The 'tools' argument is not needed here as they are already bound to the agent.
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


    async def run_agent_with_tool(query: str):
        """Invokes the agent executor with a query and prints the final response."""
        print(f"\n--- 🏃 Running Agent with Query: '{query}' ---")
        try:
            response = await agent_executor.ainvoke({"input": query})
            print("\n--- ✅ Final Agent Response ---")
            print(response["output"])
        except Exception as e:
            print(f"\n🛑 An error occurred during agent execution: {e}")

    async def main():
        """Runs all agent queries concurrently."""
        tasks = [
            run_agent_with_tool("What is the capital of France?"),
            run_agent_with_tool("What's the weather like in London?"),
            run_agent_with_tool("Tell me something about dogs.") # Should trigger the default tool response
        ]
        await asyncio.gather(*tasks) # asyncio.gather(*tasks) 부분은 세 가지 질문을 동시에(병렬로) 처리하라는 뜻

    # Removed if __name__ == "__main__" block and directly await main()
    await main()

else:
    print("\nSkipping agent execution due to LLM initialization failure.")