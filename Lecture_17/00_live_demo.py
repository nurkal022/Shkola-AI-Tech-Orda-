import requests   
from bs4 import BeautifulSoup

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from dotenv import load_dotenv
load_dotenv()


@tool
def get_quotes(topic: str) -> str:
    """Получает цитаты на заданную тему.
    Args:
        topic: Тема цитат (например, "love", "life", "success")
    """
    url = f"https://quotes.toscrape.com/tag/?q={topic}"
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "lxml")
    quotes = soup.find_all("div", class_="quote")
    result = ""
    for quote in quotes:
        text = quote.find("span", class_="text").get_text()
        author = quote.find("small", class_="author").get_text()
        result += f"Цитата: {text} - {author}\n"
    return result

@tool
def get_weather(city: str) -> str:
    """Получает погоду для города.
    Args:
        city: Город на английском (Astana, London)
    """
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    data = response.json()
    print(data)
    return f"Погода в {city}: {data['current_condition'][0]['temp_C']}°C"



llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [get_weather, get_quotes]
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты полезный AI-ассистент с инструментами для получения информации из интернета. Используй get_weather для погоды и get_quotes для цитат. Отвечай на русском языке."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])


agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Агент с инструментами готов!")
    print("=" * 60)
    print()
    result = agent_executor.invoke({"input": "Покажи мне  цитаты на тему love."})
    print(result["output"])
