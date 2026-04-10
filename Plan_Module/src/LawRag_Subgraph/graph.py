import sys
# 禁用tqdm

from Asyn_hybridSearch.Asyn_hybridSearch import hybridSearch
import json
from zerollm.zerollm import ZeroAgent
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from typing import List, TypedDict, Literal, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage,AIMessageChunk,HumanMessage,AIMessage,BaseMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import traceback
from uuid import uuid4
import asyncio
from FlagEmbedding import FlagReranker
from Asyn_hybridSearch.Asyn_hybridSearch import BGEM3Embedding


reranker_model = FlagReranker('/home/25-fengyuan/LLModel/bge-reranker-v2-m3', use_fp16=True, devices = 'cuda:5') # Setting use_fp16 to True speeds up computation with a slight performance degradation
embedding_model = BGEM3Embedding(device='cuda:5') 

class Gloabel_var(TypedDict):
    reaction: str # 示例反馈
    rerank_threshold:float
    max_generate_times:int
    max_check_point_times:int

# 混合检索时间测试
# a = time.time()
mySearch1 = hybridSearch(search_collection = 'pdf_chunck_with_metadata', embedding_model = embedding_model)
gloabel_var = Gloabel_var(reaction="", rerank_threshold=0.75, max_generate_times=3, max_check_point_times=2)

# ans = mySearch(ques = '请告诉我《关于做好民政服务机构汛期安全管理工作的通知》相关内容', spare_topk= 100, dense_topk = 100, topk = 50)# "谁发表了'与家乡相伴找到了力量，与孩子们相伴找到了快乐，爱我所爱无怨无悔，扎根边疆的好园丁'的评论" # '请查找台风鲇鱼相关报道！'
# print(f'用时:{time.time() - a}')


# 网络搜索工具
# os.environ["TAVILY_API_KEY"] = "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"
# web_search_tool = TavilySearchResults(k=3)
# Agent
myZeroAgent = ZeroAgent()
grader_llm = myZeroAgent.get_grader_llm(is_test = None, is_mine = 'Mine')
router_llm = myZeroAgent.get_router_llm(is_test=None, is_mine='Mine')
rag_llm = myZeroAgent.get_rag_llm(is_mine = 'Mine')
hallucination = myZeroAgent.get_hallucination_llm(is_mine = 'Mine')
answerQ = myZeroAgent.get_answerQ_llm(is_mine='Mine')
reWriter_llm = myZeroAgent.get_reWriteQus_llm(is_mine = 'Mine')
rethink_llm = myZeroAgent.get_hallucination_rethink(is_mine='Mine')

db_collection_choice = '' # 应该没用

class GraphState(TypedDict):
    """
    图传递元素
    Attributes:
        messages: 上下文
        improved_question: 改进后的问题
        generation: 模型的回答
        documents: 文档list
        generate_times: 生成次数
        check_point_times: 检查次数
    """
    messages: Annotated[list[BaseMessage], add_messages]
    improved_question:str
    generation: str
    documents: List[str]
    generate_times:int = 0
    check_point_times:int = 0

    # db_collection_choice: str
async def check_point(state):
    check_point_times = state.get('check_point_times', 0) + 1
    # 如果没有improved_question，则使用messages的最后一个问题作为问题
    if (not state.get('improved_question', False) ) or (state['improved_question'] == ''):
        question = state['messages'][-1].content
    else:
        # 否则使用improved_question作为问题，当前的问题用于检索
        question = state["improved_question"]
    
    # print('=====当前的上下文=====')
    # print(state['messages'])
    # input()
    # print('=====当前的上下文=====')
    return {"improved_question": question, 'check_point_times': check_point_times}
async def retrieve(state):
    """
    检索文档
    """
    question = state["improved_question"]
    print(f'question is {question}')
    documents = await mySearch1(ques = question, spare_topk= 100, dense_topk = 100, topk = 20)# "谁发表了'与家乡相伴找到了力量，与孩子们相伴找到了快乐，爱我所爱无怨无悔，扎根边疆的好园丁'的评论" # '请查找台风鲇鱼相关报道！'
    return {"documents": documents, "improved_question": question} #"generation":''

# 中间穿插一个reranker更好
async def reranker(state):
    """
    文档重排序
    """
    question = state["improved_question"]
    documents = state["documents"]
    filtered_documents = []
    for d in documents:
        document = f'标题:{d.get("title")}\n内容:{d.get("text")}'

        score = await asyncio.to_thread(reranker_model.compute_score_single_gpu, [question, d.get('raw_data')], normalize=True)
        print('====sore=====')
        print(score)
        print(question)
        print(d.get('title'),'/',d.get('page'))
        print('====sore=====')
        if score[0] > gloabel_var.get('rerank_threshold'):
            print(f'document is {d.get("title")}/{d.get('page')}: ✅,  with score {score}')
            filtered_documents.append(d)
        else:
            print(f'document is {d.get("title")}/{d.get('page')}: ❌, with score {score}')
    return {"documents": filtered_documents}



async def grade_documents(state):
    """
    重排序后的文档相关性评分
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    # 思考要不要加上下文
    question = state["improved_question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    print(documents)
    for d in documents:
        document = f'标题:{d.get("title")}\n内容:{d.get("text")}'
        score = await grader_llm.ainvoke(
            {"question": question, "document": d.get('raw_data')}
        )
        grade = json.loads(score.content).get('score')
        if grade == "yes":
            print(f'document is {d.get("title")}: ✅')
            filtered_docs.append(d)
        else:
            print(f'document is {d.get("title")}: ❌')
            continue

    return {"documents": filtered_docs, "improved_question": question}

async def generate(state):
    """
    根据文档生成回答
    """
    generate_times = state.get('generate_times', 0) + 1
    print("---GENERATE---") 
    question = state["improved_question"]
    documents = state["documents"]
    # RAG generation
    generation = await rag_llm.ainvoke({"context": documents, "question": question, "reaction":gloabel_var.get('reaction')}) # 生成纯字符
    # 上下文裁剪
    if len(state['messages']) > 20:
        state['messages'] = state['messages'][-20:]
    gloabel_var['reaction'] = ''
    return {"documents": documents, "improved_question": question, "generation": generation.content, 'generate_times': generate_times} # 'messages':[generation]

async def improve_query(state):
    """
    重新生成问题
    """

    print("---TRANSFORM QUERY---")
    # 根据6条上下文重写问题
    full_history = myZeroAgent.Message2History(state['messages'])
    # 取最后6条作为上下文，最后1条作为问题
    if len(full_history) > 1:
        history = full_history[-6:-1]  # 倒数第2到第6条作为历史
        question = full_history[-1]     # 最后一条作为当前问题
    else:
        history = []
        question = full_history[-1] if full_history else ""
    # documents = state["documents"]
    print('=======重写问题======')
    # Re-write question
    better_question = await reWriter_llm.ainvoke({"history": history, "question": question})
    print(f'history is {history}')
    print(f'question is {question}')
    # input()
    print('=======重写问题======')

    return {"documents": [], "improved_question": better_question.content}


async def route_question(state):
    """
    数据库路由
    """
    if state['check_point_times'] > gloabel_var.get('max_check_point_times'): # 最多路过check2次，第3次路过直接生成答案
        return 'refresh'
    gloabel_var['retrieve'] = mySearch1
    return "vectorstore1"
    # print("---ROUTE QUESTION---")
    # question = state["messages"][-1].content
    # # input(state)
    # source = await router_llm.ainvoke({"question": question})
    # print(source)
    # datasource = json.loads(source.content).get('datasource')
    # if datasource == "vectorstore1":
    #     print("---ROUTE QUESTION TO RAG VECTORSTORE1---")
    #     gloabel_var['retrieve'] = mySearch1
    #     return "vectorstore1"
    # elif datasource == "vectorstore2":
    #     print("---ROUTE QUESTION TO RAG VECTORSTORE2---")
    #     gloabel_var['retrieve'] = mySearch2
    #     return "vectorstore1"
    # elif datasource == "self":
    #     print("---ROUTE QUESTION TO SELF---")
    #     return "self"

async def decide_to_generate(state):
    """
    判断是否需要重新生成问题,因为无相关文档,需要重新生成问题
    """

    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]

    if not filtered_documents:# 没有一个相关的
        print(
            "---没有相关文档，需要重新生成问题---"
        )
        return "improve_query"
    else:
        # We have relevant documents, so generate answer
        print("---有相关文档可以生成回答---")
        return "generate"

async def old_grade_generation_v_documents_and_question(state):
    
    #     """
    #     判断生成内容是否符合文档和问题
    #     """
        
    #     # 获取文档， 生成内容，用户问题
    #     if (not state.get('improved_question', False) ) or (state['improved_question'] == ''):
    #         question = state["messages"][-1].content
    #     else:
    #         question = state["improved_question"]
    #     documents = state["documents"]
    #     generation = state["generation"]
    #     # messages = state['messages']

    #     # 幻觉检测获得 score和reason
    #     while True:
    #         try:
    #             score = await hallucination.ainvoke(
    #                 {"documents": documents, "generation": generation}
    #             )
    #             json_score = json.loads(score.content)
    #             grade = json_score.get('score')
    #             break
    #         except Exception as e:
    #             print(f"幻觉检测json格式生成失败: {e}")
    #             continue
        
    #     # 没有基于文档生成的内容相较文档有幻觉
    #     if grade == "no":
    #         # 再次思考
    #         while True:
    #             try:
    #                 rethink_ans = await rethink_llm.ainvoke(
    #                     {"question": question, "documents": documents, "generation": generation, "reason":json_score.get('reason')}
    #                 )
    #                 rethink_binary_score = json.loads(rethink_ans.content).get('score')
    #                 break
    #             except Exception as e:
    #                 print(f"文档幻觉监测：再次思考json格式生成失败: {e}")
    #         # 维持原判
    #         if rethink_binary_score == 'yes':
    #             gloabel_var['reaction'] = "无"
    #             if json_score.get('reason') != "":
    #                 gloabel_var['reaction'] += f"幻觉生成案例【{json_score.get('reason')}，请不要犯这样的错误\n】"
    #             return "not supported"
    #     # 否则表明符合文档
    #     print("---符合文档，继续判断是否符合问题---")
    #     # 要加历史信息---最终版的【替代方案】
    #     history = myZeroAgent.Message2History(state['messages'])
    #     # 不去除ai的回答
    #     split_ques = history
    #     # split_ques = history[-6:-1]# 要去除ai的回答+
    #     # input(split_ques)
    #     documents = state["documents"]
    #     # 根据上下文生成一个更好的问题
    #     better_question = await reWriter_llm.ainvoke({"question": split_ques})

    #     while True:
    #         try:
    #             score = await answerQ.ainvoke({"question": better_question, "generation": generation})
    #             grade = json.loads(score.content).get('score')
    #             reason = json.loads(score.content).get('reason')
    #             break
    #         except Exception as e:
    #             print(f"回答问题json格式生成失败: {e}")
    #             continue
    #     if grade == "yes":
    #         print("没有任何幻觉")
    #         return "useful"
    #     else:
    #         # 
    #         rethink_ans = await rethink_llm.ainvoke(
    #             {"question": question, "documents": documents, "generation": generation, "reason":reason}
    #         )
    #         rethink_binary_score = json.loads(rethink_ans.content).get('score')
    #         # 维持原判
    #         if rethink_binary_score == 'yes':
    #             if reason:
    #                 gloabel_var['reaction'] += f"幻觉生成案例【{json_score.get('reason')}，请不要犯这样的错误\n】"
    #             print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
    #             return "not useful"
    #     # 那么就行了
    #     return "useful"
    pass
async def grade_generation_v_documents_and_question(state):
    """
    判断生成内容是否符合文档和问题
    """
    
    if state['generate_times'] >= gloabel_var.get('max_generate_times'):
        state['generation'] = '问题可能尚未完全解决，这是目前能尽力给出的答案：' + state['generation']
        return "useful"
    # 获取文档， 生成内容，用户问题
    if (not state.get('improved_question', False) ) or (state['improved_question'] == ''):
        question = state["messages"][-1].content
    else:
        question = state["improved_question"]
    documents = state["documents"]
    generation = state["generation"]
    # messages = state['messages']

    # 幻觉检测获得 score和reason
    while True:
        try:
            score = await hallucination.ainvoke(
                {"documents": documents, "generation": generation}
            )
            json_score = json.loads(score.content)
            grade = json_score.get('score')
            break
        except Exception as e:
            print(f"幻觉检测json格式生成失败: {e}")
            continue
    
    is_ok = True
    # 没有基于文档生成的内容相较文档有幻觉
    if grade == "no":
        print("回答与文档矛盾")
        is_ok = False
        if json_score.get('reason') != "":
            gloabel_var['reaction'] += f"幻觉生成案例【{json_score.get('reason')}，请不要犯这样的错误】\n"
        return "not supported"
    print("---继续判断是否符合问题---")
    # 要加历史信息---最终版的【替代方案】
    history = myZeroAgent.Message2History(state['messages'])
    # split_ques = history[-6:-1]# 要去除ai的回答+
    # input(split_ques)
    documents = state["documents"]
    # 根据上下文生成一个更好的问题
    better_question = await reWriter_llm.ainvoke({"history": history[:-1],"question": history[-1]}) # split_ques
    while True:
        try:
            score = await answerQ.ainvoke({"question": better_question, "generation": generation})
            grade = json.loads(score.content).get('score')
            reason = json.loads(score.content).get('reason')
            gloabel_var['reaction'] += f"幻觉生成案例【用户问题:{better_question}, 模型回答: {generation}, 这是一个错误案例，因为：{reason}，请不要犯这样的错误】\n"
            break
        except Exception as e:
            print(f"回答问题json格式生成失败: {e}")
            continue
    if grade == "no":
        print("回答与问题无关")
        is_ok = False
    if is_ok:
        return "useful"
    else:
        return "not useful"


# 收尾工作
def clear_tmp_varible(state):
    # 添加ai 最后的回答
    if state.get('generation','') == '':
        state['generation'] = '并未在数据库中检索到相关知识！'
    final_generation = state['generation'][:]
    return {'messages':[AIMessage(content=final_generation)], 'improved_question':'', 'generation':'','documents':[], 'reaction': '', 'generate_times': 0, 'check_point_times': 0}
# 构建图
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # 检索器
workflow.add_node("grade_documents", grade_documents)  # 文档相关性评分
workflow.add_node("generate", generate)  # 根据文档生成内容
workflow.add_node("improve_query", improve_query)  # 问题精炼
workflow.add_node("refresh", clear_tmp_varible)
workflow.add_node("check_point", check_point)
workflow.add_node("make_better_query", improve_query)
workflow.add_node("reranker", reranker)

# Build graph
workflow.add_edge(START, "check_point")
workflow.add_edge("make_better_query", "check_point")
workflow.add_conditional_edges(
    "check_point",
    route_question,
    {
        "vectorstore1": "retrieve",
        "refresh": "refresh",
    },
)
workflow.add_edge("retrieve", "reranker")
workflow.add_edge("reranker", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "improve_query": "improve_query",
        "generate": "generate",
    },
)
workflow.add_edge("improve_query", "check_point")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": "refresh",
        "not useful": "make_better_query",
    },
)
workflow.add_edge("refresh", END)

# Compile
Rag_agent = workflow.compile()




