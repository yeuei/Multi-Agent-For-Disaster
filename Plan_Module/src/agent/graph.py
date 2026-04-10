import json
from qwen_model import get_llm, StructureAgent, draw_flow
from langgraph.graph import StateGraph, START, END, MessagesState, add_messages
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, RemoveMessage, BaseMessage
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
import random
import re
import asyncio
import aiofiles
from qwen_model import Route2Agent, Summarize_History
from functools import wraps
import time

# from langchain_core.messages import convert_to_messages
# from langchain_core.messages import messages_to_dict


# 试错包装器
def retry(retries: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,), on_retry=None):
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exc = None
                for attempt in range(1, retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exc = e
                        if on_retry:
                            try:
                                on_retry(attempt, e)
                            except Exception:
                                pass
                        if attempt == retries:
                            raise
                        await asyncio.sleep(delay)
                raise last_exc
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exc = None
                for attempt in range(1, retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exc = e
                        if on_retry:
                            try:
                                on_retry(attempt, e)
                            except Exception:
                                pass
                        if attempt == retries:
                            raise
                        time.sleep(delay)
                raise last_exc
            return sync_wrapper
    return decorator




def make_history(message_list: list[BaseMessage]) -> str:
    custom_message_dicts = []
    for message in message_list:
        custom_message_dicts.append({"role": message.type, "content": message.content})
    return str(custom_message_dicts)


# 初始化 executor_llm 和 give_answer_llm

executor_llm = get_llm() 
give_answer_llm = get_llm()

# 引入子图中的workflows
from Document_Subgraph import Document_agent as document_llm
from LawRag_Subgraph import Rag_agent as law_llm
from Knowledge_Subgraph import Knowledge_agent as knowledge_llm
from WebSearch_Subgraph import WebSearch_agent as websearch_llm
from Emergency_Subgraph import Emergency_agent as emergency_llm


plan_save_path = 'plan.md'
# max_try 最大专家交互次数
max_try = 4

qwen_llm = get_llm(temperature = 0)
class MyState(TypedDict):
    pre_messages: Annotated[list[BaseMessage], add_messages] = []
    is_makeplan_or_finishone: bool = True # True: make plan, False: finish one plan
    go_to: str = 'make_plan'
    plan_list: dict[str, str] = {}
    exist_info: str = ''
class executor_state(MyState):
    exe_message: Annotated[list[BaseMessage], add_messages] = []
    ask_expert_times: int = 0 # 专家交互次数
    is_real_answer: bool = False # 是否是正确回答
    point_answer: str = '' # 答案
    question: str = '' # 总任务
    single_question: str = '' # 单个子任务


# plan_agent = StructureAgent(qwen_llm, plan_prompt, Plan_structure)

def init(state: MyState):
    print('=========当前上下文+=========')
    print(f'pre_messages: {state["pre_messages"]}')
    print('=========当前上下文-=========')
    return{
        'is_makeplan_or_finishone': True,
        'go_to': 'make_plan',
        # 接受用户的提问
        # 'pre_messages': state['pre_messages']
    }

async def make_plan(state: MyState):
    async with aiofiles.open('./prompt/make_plan.txt', 'r') as f:
        plan_prompt = await f.read()
    system_prompt = SystemMessage(content=plan_prompt)
    history = [system_prompt] + state['pre_messages']
    plan_list = await qwen_llm.ainvoke(history)
    # 匹配 ```json```
    
    # 尝试3次
    times = 0
    while True:
        try:
            plan_json = re.search(r'```json(.*)```', plan_list.content, re.DOTALL).group(1).strip()
            plan_list = json.loads(plan_json)
            break
        except:
            times += 1
            if times >= 3:
                raise Exception('make plan failed')
            plan_list = await qwen_llm.ainvoke(history)
    return{'is_makeplan_or_finishone': True,
            'plan_list': plan_list,
            'go_to': 'make_planfiles'}
def check_make_planfiles(state: executor_state):
    if state['go_to'] == 'executor':
        return 'executor'
    else:
        return 'final_response'

async def make_planfiles(state: executor_state):
    if state['is_makeplan_or_finishone']:
        print(f'==================================第一次制定计划==================================')
        # 第一次制定计划
        plan_list = state['plan_list']
        async with aiofiles.open(plan_save_path, 'w') as f:
            for key, value in plan_list.items():
                await f.write(f"[ ] {value}\n")
        state['exe_message'] = []

        # 获取第一个任务
        async with aiofiles.open(plan_save_path, 'r') as f:
            first_question = (await f.readline())
        first_question = first_question.split(']')[1].strip()
        return{
            'is_makeplan_or_finishone': False, # 不再需要制定计划
            'go_to': 'executor', # 转到执行计划
            # 初始化上下文
            'ask_expert_times': 0,
            'is_real_answer': False,
            'question': first_question,
        }
    else:
        print(f'==================================继续执行计划==================================')
        # 继续执行计划
        async with aiofiles.open(plan_save_path, 'r') as f:
            rows = await f.readlines()
        print(f'rows: {rows}')
        # input('查看当前计划列表')
        # lens = len(rows)
        for idx, row in enumerate(rows):
            if row.startswith('[ ]'):
                is_leaft_plan = True
                if state['is_real_answer']:
                    new_row = f'[✅]{row.split(']')[1].strip()}-->回答:{state['point_answer']}\n'
                else:
                    new_row = f'[❌]{row.split(']')[1].strip()}-->回答:{state['point_answer']}\n'
                rows[idx] = new_row
                break
        is_leaft_plan = False
        next_question = ''
        async with aiofiles.open(plan_save_path, 'w') as f:
            for row in rows:
                if not is_leaft_plan and row.startswith('[ ]'):
                    is_leaft_plan = True
                    next_question = row.split(']')[1].strip()
                await f.write(row)
        if is_leaft_plan:
            # 还有任务
            # 清空上下文
            # state['exe_message'] = []
            return{
                'is_makeplan_or_finishone': False,
                'go_to': 'executor',
                # 初始化上下文
                'ask_expert_times': 0,
                'is_real_answer': False,
                'question': next_question,
                'exe_message': RemoveMessage(id="__remove_all__"),
            }
        else:
            # 没有任务
            return{
                'is_makeplan_or_finishone': False,
                'go_to': 'final_response'
            }
async def give_answer(state: executor_state):
    # 测试
    print(f'当前的问题是:{state["question"]}')
    async with aiofiles.open('./prompt/SummarizeHistory.txt', 'r') as f:
        give_answer_prompt = await f.read()
        
    give_answer_llm_struct = StructureAgent(give_answer_llm, give_answer_prompt, Summarize_History)
    history = make_history(state['exe_message'])
    print('==============================')
    print(f'history: {history}')
    print('==============================')

    new_prompt = give_answer_llm_struct.sys_prompt.format(state['question'], history)
    final_answer = await give_answer_llm_struct.llm_struture.ainvoke(new_prompt)
    is_ok = final_answer.is_ok
    point_answer = final_answer.information
    return {
        'is_real_answer': is_ok == 'yes',
        'point_answer': point_answer,
        'go_to': 'make_planfiles',
    }
    # if random.random() < 0.5:
    #     print(f'give_answer:问题很简单无需额外信息')
    #     return{
    #         'is_real_answer': True,
    #         'point_answer': '问题很简单无需额外信息',
    #         'go_to': 'make_planfiles'
    #     }
    # else:
    #     print(f'give_answer:我无法回答这个问题')
    #     return{
    #         'is_real_answer': False,
    #         'point_answer': '我无法回答这个问题',
    #         'go_to': 'make_planfiles'
    #     }

async def direct_websearch(state: executor_state):
    
    exist_info = f'任务完成进度:{state["exist_info"]} \n\n---------\n\n'
    web_ans = await websearch_llm.ainvoke({"messages": [{"role": "user", "content": exist_info + state['single_question']}]})
    web_ans = web_ans.get('messages', '')[-1]
    web_ans.content = 'websearch_agent的回复：' + web_ans.content 
    return{
        'go_to': 'executor',
        'exe_message': web_ans
    }
async def law_agent(state: executor_state):
    # exist_info = f'任务完成进度:{state["exist_info"]} \n\n---------\n\n'
    lawrag_ans = await law_llm.ainvoke({"messages": state['single_question']})
    lawrag_ans = lawrag_ans.get('messages', '')[-1]
    lawrag_ans.content = 'lawrag_agent的回复：' + lawrag_ans.content  
    return{
        'go_to': 'executor',
        'exe_message': lawrag_ans
    }
async def knowledge_agent(state: executor_state):
    exist_info = f'任务完成进度:{state["exist_info"]} \n\n---------\n\n'
    knowledge_ans = await knowledge_llm.ainvoke({"messages": exist_info + state['single_question']})
    knowledge_ans = knowledge_ans.get('messages', '')[-1]
    knowledge_ans.content = 'knowledge_agent的回复：' + knowledge_ans.content 
    return{
        'go_to': 'executor',
        'exe_message': knowledge_ans
    }
async def document_agent(state: executor_state):
    exist_info = f'已知信息:{state["exist_info"]} \n\n---------\n\n交给你的任务:'
    document_ans = await document_llm.ainvoke({"messages": [{"role": "user", "content": exist_info + state['single_question']}]})
    document_ans = document_ans.get('messages', '')[-1]
    document_ans.content = 'document_agent的回复：' + document_ans.content 
    return{
        'go_to': 'executor',
        'exe_message': document_ans
    }
async def emergency_agent(state: executor_state):
    exist_info = f'任务完成进度:{state["exist_info"]} \n\n---------\n\n'
    emergency_ans = await emergency_llm.ainvoke({"messages": exist_info + state['single_question']})
    emergency_ans = emergency_ans.get('messages', '')[-1]
    emergency_ans.content = 'emergency_agent的回复：' + emergency_ans.content 
    return{
        'go_to': 'executor',
        'exe_message': emergency_ans
    }

def check_executor(state: executor_state):
    if state.get('ask_expert_times', 0) >= max_try:
        return 'give_answer'
    if state.get('go_to', 'None') == 'finish':
        return 'give_answer'
    else:
        return state['go_to'] # law_agent, knowledge_agent, websearch_agent, document_agent, emergency_agent

async def executor(state: executor_state):
    async with aiofiles.open('./prompt/route2agent.txt', 'r') as f:
        executor_prompt = await f.read()
    exist_info = ''
    async with aiofiles.open(plan_save_path, 'r') as f:
        rows = await f.readlines()
        for idx, row in enumerate(rows):
            if row.startswith('[ ]'):
                break
            exist_info += row + '\n'
    

    executor_llm_struct = StructureAgent(executor_llm, executor_prompt, Route2Agent)

    history = make_history(state['exe_message'])

    new_prompt = executor_llm_struct.sys_prompt.format(exist_info, state['question'], history)

    decision_executor = await executor_llm_struct.llm_struture.ainvoke(new_prompt)
    
    single_question = decision_executor.question
    go_to = decision_executor.go_to
    return {'go_to': go_to,
            'single_question': single_question,
            'exe_message': AIMessage(content = f'总管代理 -> {go_to}:\n'+ single_question),
            'ask_expert_times': state['ask_expert_times'] + 1,
            'exist_info': exist_info
            }

async def final_response(state: executor_state):
    async  with aiofiles.open('./prompt/final_response.txt', 'r') as f:
        final_response_prompt = await f.read()
    system_prompt = SystemMessage(content=final_response_prompt)
    async with aiofiles.open(plan_save_path, 'r') as f:
        plan = await f.read()
    assistant_prompt = AIMessage(content='我是一位可靠的助手，这是关于计划清单的详细解答:\n' + plan)
    history = [system_prompt] + state['pre_messages'] + [assistant_prompt]
    ans = await qwen_llm.ainvoke(history)
    return {
        'pre_messages': [assistant_prompt] + [ans],
    }
# 对话记忆存储点
memory = MemorySaver()
# 对话记忆线程
config = {"recursion_limit": 1000, "configurable": {"thread_id": "1"}}

# 构件图
app = StateGraph(MyState)
app.add_node('init', init)
app.add_node('make_plan', make_plan)
app.add_node('make_planfiles', make_planfiles)
app.add_node('executor', executor)
app.add_node('give_answer', give_answer)
app.add_node('final_response', final_response)
app.add_node('websearch_agent', direct_websearch)
app.add_node('law_agent', law_agent)
app.add_node('knowledge_agent', knowledge_agent)
app.add_node('document_agent', document_agent)
app.add_node('emergency_agent', emergency_agent)


app.add_edge(START, 'init')
app.add_edge('init', 'make_plan')
app.add_edge('make_plan', 'make_planfiles')
app.add_conditional_edges('make_planfiles', check_make_planfiles, path_map = {'executor': 'executor', 'final_response': 'final_response'})
app.add_conditional_edges('executor', check_executor, path_map = {'give_answer': 'give_answer', 'websearch_agent': 'websearch_agent', 'law_agent': 'law_agent', 'knowledge_agent': 'knowledge_agent', 'document_agent': 'document_agent', 'emergency_agent': 'emergency_agent'}) # law_agent, knowledge_agent, websearch_agent, document_agent, emergency_agent
app.add_edge('websearch_agent', 'executor')
app.add_edge('law_agent', 'executor')
app.add_edge('knowledge_agent', 'executor')
app.add_edge('document_agent', 'executor')
app.add_edge('emergency_agent', 'executor')
app.add_edge('give_answer', 'make_planfiles')
app.add_edge('final_response', END)

# graph = app.compile(checkpointer=memory)
graph = app.compile()