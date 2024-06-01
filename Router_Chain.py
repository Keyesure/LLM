from typing import Union

from langchain.agents import create_react_agent, AgentExecutor
from langchain.chains.llm_math.base import LLMMathChain
from langchain.chat_models import ChatOpenAI
import os
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.python import PythonREPL
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.tools import Tool, BaseTool
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE as RounterTemplate
from langchain.chains import ConversationChain
from langchain.chains.router import MultiPromptChain
from MyOpenAIKey import getNew, getTavilyKey

# 模型
os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 数据库
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

########################
os.environ["TAVILY_API_KEY"] = getTavilyKey()

react_prompt = PromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'],
    template=
    """Assistant 是经过大量数据训练的大型语言模型。Assistant 
    的设计目的是能够协助完成各种任务，从回答简单的问题到提供对广泛主题的深入解释和讨论。作为语言模型，Assistant
    能够根据收到的输入生成类似人类的文本，使其能够进行听起来很自然的对话，并提供与当前主题相关且连贯的响应。Assistant 
    不断学习和改进，其功能也在不断发展。它能够处理和理解大量文本，并可以利用这些知识对各种问题提供准确和翔实的回答。此外，Assistant
    能够根据收到的输入生成自己的文本，使其能够参与讨论，并提供关于各种主题的解释和描述。总的来说，Assistant
    是一个功能强大的工具，可以帮助完成各种任务，并提供有关各种主题的有价值的见解和信息。无论是需要针对特定问题提供帮助，还是只想就某个特定话题进行对话，Assistant 
    都可以提供帮助。你的任务是回答用户向你提出的关于各个领域的问题，你非常擅长分析他们提出的问题，并且擅长使用工具和分析工具给出的答案。  
    
    你可以使用如下工具：
    {tools}
    
    回答问题时使用以下格式：

    Question：现在要回答的问题

    Thought：你总是要思考该如何去解决问题。

    Action：所采取的行动，需要是下面其中之一[{tool_names}]

    Action Input：行动的输入

    Observation：行动的输出

    (...以上 Thought/Action/Action Input/Observation 的过程将重复执行数遍)

    Thought：完成了所有的步骤，这个答案符合要求，我知道最终答案了。

    Final Answer：原始问题的最终答案以及合理的解释


    开始吧！
    
    Question: {input}
    
    让我们将问题分解为多个步骤，一步一步地通过(Thought/Action/Action Input/Observation)的过程解决问题以确保最终答案是正确的。
    
    Thought: {agent_scratchpad}
"""
)

llm_prompt = PromptTemplate(
    input_variables=["input"],
    template="你是一个负责解决问题的推理工具，回答用户基于逻辑的问题。"
             "从逻辑上得出解决方案，并且这个方案是现实的。"
             "在您的答案中，清楚地详细说明所涉及的步骤，并给出最终答案。\n"
             "如果输入中没有具体的问题，请根据输入与其进行普通对话"
             "Question: {input} \n"
             "Answer: 你的回答\n"
)

llm_math_prompt = PromptTemplate(
    input_variables=["input"],
    template="你可以熟练地生成一段Python代码来解决下面的问题, 要求在最后得到的参数使用print()函数"
             "{input},你的回答只包含代码",
)
llm_chain = LLMChain(llm=llm, prompt=llm_prompt)
MathChain = LLMChain(llm=llm, prompt=llm_math_prompt)
search = TavilySearchAPIWrapper()
tavily = TavilySearchResults(api_wrapper=search)
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


class CalculatorTool(BaseTool):
    name = "CalculatorTool"
    description = """
        当需要回答关于数学问题或是需要运行Python程序的问题时，你可以使用这个工具
        """

    def _run(self, question: str):
        try:
            response = MathChain.run(question)
            res = PythonREPL()
            if response[0:9] == "```python":
                output = response[9:-3]
                return res.run(output)
            else:
                return res.run(response)
        except:
            return "这个工具无法得到有效的答案，请尝试其他工具。"

    def _arun(self, value: Union[int, float]):
        raise NotImplementedError("This tool does not support async")


cal = CalculatorTool()

tools = [
    Tool(
        name='逻辑推理工具',
        func=llm_chain.run,
        description='使用该工具进行逻辑推理或是文本生成，也可以用来进行普通的对话，不使用该工具进行搜索查找'
    ),
    Tool(
        name="计算器",
        description="当需要回答关于数学问题或是需要运行Python程序的问题时，你可以使用这个工具",
        func=cal.run,
    ),
    Tool(
        name="维基百科",
        func=wikipedia.run,
        description="当你想知道某件事物的具体描述和背景细节时可以使用此工具"
    ),
    Tool(
        name='TavilySearch',
        description='这是一个搜索引擎工具，可以使用这个工具在线查找一些最近的信息，也可以搜索你想知道的任何信息',
        func=tavily.run,
    ),
]

# initialize agent with tools
react_agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
)
##########################

sql_prompt = PromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'],
    template=
    """
    你是一个专门用来解决数据库查询问题的人工智能助手，你可以将用户向你提出的问题转化为SQL，并尝试查询数据库，
    当然，你每次生成的SQL并不一定符合数据库内的实际情况，所以你可以积极尝试使用下面所提供的工具来不断纠正，
    你很有可能遇到需要跨表查询的问题，请根据获取到的数据库表信息认真发现各个表之间的联系。

    有以下工具可供使用：

    {tools}

    现在你已经知道你可以使用的工具了。

    回答问题时使用以下格式：

    Question：现在要回答的问题

    Thought：应当思考该如何去做,请分析问题，一步一步地解决问题

    Action：所采取的行动，可以是下面其中之一[{tool_names}]

    Action Input：行动的输入

    Observation：行动的输出

    (...以上 Thought/Action/Action Input/Observation 的过程将重复执行多次)

    Thought：我知道最终答案了。

    Final Answer：原始问题的最终答案

    以上便是你所要遵循的格式。

    开始吧！


    Question: {input}

    Thought: {agent_scratchpad}
"""
)

sql_agent = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    prompt=sql_prompt,
    handle_parsing_errors=True
)
##############################

# 构建提示信息
prompt_infos = [
    {
        "key": "normal_agent",
        "description": "适合使用工具回答逻辑推理、常识、运算、信息提取等有关的问题，不参与数据库的问题",
    },
    {
        "key": "sql_agent",
        "description": "这个链用来执行数据库查询，适合回答与下列几个表有关的问题"
                    + str(db.get_usable_table_names())
    }]

# 构建目标链
chain_map = {"sql_agent": sql_agent, "normal_agent": agent_executor}

# 构建路由链

destinations = [f"{p['key']}: {p['description']}" for p in prompt_infos]
router_template = RounterTemplate.format(destinations="\n".join(destinations))
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(), )
router_chain = LLMRouterChain.from_llm(llm,
                                       router_prompt,
                                       verbose=True)

# 构建默认链
default_chain = ConversationChain(llm=llm,
                                  output_key="text",
                                  verbose=True)

# 构建多提示链
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=chain_map,
    default_chain=default_chain,
    verbose=True)

chain.run("把大象关进冰箱需要几步？")
