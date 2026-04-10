import sys, os
import traceback

# 增加一个父级目录，在该目录下寻找enssential_module

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import json
import time
from langchain_core.output_parsers import StrOutputParser
from typing import Any, Literal
from abc import abstractmethod
from langchain_core.messages import AIMessage, HumanMessage
import re
from pydantic import BaseModel, Field
from qwen_model.qwen_model import get_llm as get_client
from langchain_core.tools import tool
# 路由llm结构化输出
# structured_llm_router = structured_llm_router.with_structured_output(RouteQuery)


# 定义Router的格式化输出，本质是0样本代理
# class RouteQuery(BaseModel):
#     """将用户查询路由到最相关的数据源"""
#     datasource:Literal["vectorstore1", "vectorestore2", "web_search"] = Field(default=..., description="给定一个用户问题，选择将其路由到网络搜索或向量存储。")



class ZeroAgentBase():
    def clear_json(text:str):
      return  re.match(r'.*?(\{.+\}).*?', text, flags=re.DOTALL).group(1)
    def __init__(self):
        pass
    # @abstractmethod
    # def draw_geo(self, region_dict, discribe = '受伤人数', region = '重庆'):
    #     pass
    @abstractmethod
    def get_router_llm(self, is_test = None, is_mine = None, is_json = True)->Any:
        """
        方式路由Agent
        :param is_json:
        :param is_test: 是否测试
        :param is_mine: 是否使用本地部署的llm
        :return: 返回openai格式兼容的大模型
        """
        pass
    @abstractmethod
    def get_grader_llm(self, is_test = None, is_mine = None)->Any:
        """
        检索到的文档是否相关的Agent
        :param is_test: 是否测试
        :param is_mine: 是否使用本地部署的llm
        :return: 返回openai格式兼容的大模型
        """
        pass
    def get_rag_llm(self, is_test = None, is_mine = None)->Any:
        """
        根据文档生成回答的Aent
        :param is_test: 是否测试
        :param is_mine: 是否使用本地部署的llm
        :return: 返回openai格式兼容的大模型,但是该模型的invoke的输出是str需要注意一下
        """
        pass
    @abstractmethod
    def get_hallucination_llm(self, is_test = None, is_mine = None)->Any:
        """
        rag_llm生成的内容是否忠实于文档的幻觉检测Agent
        :param is_test: 是否测试
        :param is_mine: 是否使用本地部署的llm
        :return: 返回openai格式兼容的大模型
        """
        pass
    @abstractmethod
    def get_hallucination_rethink(self, is_test = None, is_mine = None)->Any:
        """
        :param is_test:
        :param is_mine:
        :return:
        """
        pass
    @abstractmethod
    def get_answerQ_llm(self, is_test=None, is_mine=None)->Any:
        """
        rag_llm生成的内容是否能够回答question的检测Agent
        :param is_test: 是否测试
        :param is_mine: 是否使用本地部署的llm
        :return: 返回openai格式兼容的大模型
        """
        pass
    @abstractmethod
    def get_reWriteQus_llm(self, is_test=None, is_mine=None)->Any:
        """
        重写问题
        :param is_test: 是否测试
        :param is_mine: 是否使用本地部署的llm
        :return: 返回openai格式兼容的大模型
        """
        pass
    @abstractmethod
    def get_basic_llm(self, is_test = None, is_mine = None)->Any:
        pass
    @abstractmethod
    def Message2History(self, Messages: list[AIMessage]) -> list:
        """
        :param Messages:含有AIMessage和HumanMessage的历史对话信息
        :return: 转换成list[tuple]格式的历史对话信息
        """
        pass
class Executor_Router(BaseModel):
    go_to: Literal['law_agent', 'knowledge_agent', 'websearch_agent', 'document_agent', 'emergency_agent', 'give_answer']

class ZeroAgent(ZeroAgentBase):
    def get_grader_llm(self, is_test = None, is_mine = None):
        structured_llm_grader = get_client(base_url = 'http://127.0.0.1:8501/v1', model_name = 'Qwen2.5-7B-Instruct')
        system1 = """
            1. 主体事件匹配：
               - 灾害名称完全一致（含标点符号）
               - 或同类灾害类型（如台风≈飓风）
            
            2. 子主题泛化匹配：
               a) 地理包容原则：
                  - 文档提及省级行政区即覆盖全省所有市县（如"台湾"包含所有台湾地区）
                  - 灾害影响动词映射（横扫/影响/袭击→登陆）
               
               b) 语义扩散原则：
                  - 接受模糊时间匹配（如"27日晚"→"9月下旬"）
                  - 接受数据维度关联（伤亡人数→经济损失）
            
            # 强制yes条件（满足任一）：
            ✅ 文档与用户问题存在任意维度关联点：
               - 事件主体名称匹配
               - 子主题关键词/语义关联
               - 同类事件类型关联
            
            # 输出规范
            {{
              "score": "yes",  # 仅当完全无关联时允许no
              "reason": "[激活的关联维度]: [具体关联要素]"
            }}
            # 格式禁令：
               - 禁止添加注释/说明文字
               - 禁止使用```json等标记
               - 禁止修改字段名称大小写
               - 禁止包含非ASCII字符
            # 终极宽松案例库
            <案例1 必须yes>
            用户问题: 查找台风“鲇鱼”登陆菲律宾、浙江、台湾的报道
            检索文档内容: 台风“鲇鱼”影响台湾基隆港
            输出:
            {{
              "score":"yes",
              "reason":"地理包容原则: 基隆港∈台湾 + 动词映射:影响→登陆"
            }}
            
            <案例2 必须yes>
            用户问题: 统计地震"青龙"在日本、菲律宾的伤亡
            检索文档内容: 地震"青龙"引发东京建筑摇晃
            输出:
            {{
              "score":"yes",
              "reason":"行政隶属:东京∈日本 + 语义关联:建筑摇晃→伤亡统计"
            }}
            
            <唯一no条件案例>
            用户问题: 查找台风“鲇鱼”在亚洲的受灾报告
            检索文档内容: 飓风“凯瑟琳”影响美国佛罗里达
            输出: 
            {{
              "score":"no",
              "reason":"事件主体不匹配（鲇鱼≠凯瑟琳） + 地域无关联（亚洲≠美洲）"
            }}
            """
        system2 = """
        # 角色定位
        您是多维度主题验证评分器，严格遵循事件主体→子主题的递进验证逻辑，具备逻辑链分析能力的文档相关性评分器，必须按照以下思考框架运作：
        
        1. 一级验证（事件主体）：
           - 精确匹配用户问题中定义的事件标识（如「海啸"先锋"」）
           - 要求：名称全称、符号格式完全一致（如全角引号）
           - 不匹配 → 立即返回{{"score":"no"}}
        
        2. 二级验证（子主题群）：
           a) 解析用户问题中的子主题集合（如案例中的日本、浙江和美国）
           b) 检测文档是否包含任一子主题的：
              - 直接表述（如灾害影响地、参与组织等）
              - 语义关联（如"台风登陆"→"风力发电受损"关联能源主题）
              - 时间包含（如文档时间段包含用户指定时间范围）
           c) 不需要完全覆盖子主题 
        
        # 动态主题处理规则
        - 子主题类型自动识别：地理/时间/组织/事件类型等
        - 匹配逻辑：文档只需命中任意1个用户定义的子主题类别
        
        # 输出规范
        {{
          "score": "yes/no",
          "reason": "[事件主体状态] → [子主题匹配状态] → 结论"
        }}
        
        # 输出规范
        必须生成包含推理链的JSON对象：
        {{
          "score": "yes/no",
          "reason": "文档内容涉及[检出主题]/未提及[未检出主题] → 结论"
        }}
        
        # 验证案例库
        <案例1 应判yes>
        用户问题：请查找海啸“先锋”登陆日本、浙江和美国的报道
        检索文档内容：海啸“先锋”横扫日本已致50伤
        输出：
        {{
          "score":"yes",
          "reason":"文档内容涉及海啸“先锋”,大主题匹配→文档子主题涉及日本（横扫日本等效登陆日本），未提及浙江和美国 → 至少1个子主题匹配 → yes"
        }}
        
        <案例2 应判no>
        用户问题：请查找海啸“先锋”登陆日本、浙江和美国的报道
        检索文档内容：海啸“先锋”冲击韩国
        输出：
        {{
          "score":"no", 
          "reason":"文档内容涉及海啸“先锋”,大主题匹配→文档子主题涉及韩国，未提及日本、浙江和美国的关系 → 零主题匹配 → no"
        }}
        
        <案例3 应判no>
        用户问题：请查找海啸“先锋”登陆日本、浙江和美国的报道
        检索文档内容：泥石流“滚浪”冲击日本
        输出：
        {{
          "score":"no", 
          "reason":"文档内容涉及泥石流“滚浪”, 问题大主题海啸“先锋” → 大主题不匹配 → no"
        }}
        # 格式铁律
        - 字段名称严格保持小写
        - 推理链使用箭头符号"→"连接
        - 地理主题用括号标注检出状态
        # 接下来请处理真实例子
    """
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system1),
                ("human", "检索文档内容: \n\n {document} \n\n 用户问题: {question} 输出:"),
            ]
        )
        grade_llm = grade_prompt | structured_llm_grader
        # 测试
        tests = [{'q':"请介绍一下’海贝思‘引起的泥石流相关事件。", 'd': "'海贝思'是一款优质的奶粉"},
                 {'q':"最近丁真干了什么事情?",'d':"丁真最近在理塘放牛!"},
                 {'q':"重庆市石井坡附近发生过哪些崩塌？",'d':"崩塌 石井坡街道前进坡161、162号危岩 野外编号：187 省:重庆市 乡镇:石井坡街道 经度106.436389 纬度29.595278"}
                 ]
        if is_test in ['test', 'yes', 'YES', 'y', 'Y']:
            a = time.time()
            for test in tests:
                ans = grade_llm.invoke(
                    {"question": test['q'], 'document': test['d']}
                )
                print([ans])
                print(type(ans))
                route_dict = json.loads(ans.content)
                print(route_dict.get('score', None))
            print(f'生成{len(tests)}个回答,总共耗时{time.time() - a}s')
        return grade_llm
    def get_rag_llm(self, is_test = None, is_mine = None):
        # 基于RAG生成内容的Agent
        llm = get_client(base_url = 'http://127.0.0.1:8501/v1', model_name = 'Qwen2.5-7B-Instruct')
        system = """
        您是一位问答助手，请根据上下文回答问题，同时在生成内容的时候要考虑到专家建议：
        上下文：{context}
        问题：{question}
        专家建议:{reaction}
        回答要求：
        1. 仅使用中文
        2. 回答尽量详细但是不能过于冗长
        3. 无关问题回答"不知道"
        请注意不能犯以下3个错误:
        ⛔ 回答关键信息超出文档范围
        ⛔ 核心结论无文档依据  
        ⛔ 存在文档冲突内容
        """
        system_prompt = ChatPromptTemplate.from_messages(
            [
                ('system', system)
            ]
        )
        # 记得StrOutputParser用()实例化
        rag_llm = system_prompt | llm # | StrOutputParser()

        if is_test in ['test', 'yes', 'YES', 'y', 'Y']:
            tests = [{'q': "请介绍一下’海贝思‘引起的泥石流相关事件。", 'd': "'海贝思'是一款优质的奶粉"},
                     {'q': "最近丁真干了什么事情?", 'd': "丁真最近在理塘放牛!"},
                     {'q': "重庆市石井坡附近发生过哪些崩塌？",
                      'd': "崩塌 石井坡街道前进坡161、162号危岩 野外编号：187 省:重庆市 乡镇:石井坡街道 经度106.436389 纬度29.595278"}
                     ]

            a = time.time()
            for test in tests:
                ans = rag_llm.invoke(
                    {"question": test['q'], 'context': test['d']}
                )
                print(ans)
                print(ans.content)
            print(f'生成{len(tests)}个回答,总共耗时{time.time() - a}s')
        return rag_llm

    def get_hallucination_llm(self, is_test = None, is_mine = None):
        # 幻觉检测Agent
        structured_llm_grader = get_client(base_url = 'http://127.0.0.1:8501/v1', model_name = 'Qwen2.5-7B-Instruct')
        system = """
            # 【角色】
            事实核查专家，严格判断回答与给定文档的逻辑支持关系
            # 【判断规则】
            同时满足以下条件时输出"yes"：
            1.回答中≥50%关键信息在文档中有对应描述
            2.核心结论有文档内容直接支持
            3.不存在与文档矛盾的陈述
            任一条件成立则输出"no"：
            ⛔ 回答关键信息胡编乱造并且和文档毫无关联
            ⛔ 核心结论无文档依据  
            ⛔ 存在文档冲突内容
            {{"score":"yes", "reason":""}} 或 {{"score":"no","reason":"核心结论无文档依据，因为..."}}
            # 【示例】
                示例1（文档冲突 - no）：
                文档：量子计算机尚未实现商用化

                回答：量子计算机已投入商业使用

                核查专家：{{"score":"no", "reason":"存在文档冲突内容，文档说'量子计算机尚未实现商用化'，表明未商用；但回答说'量子计算机已投入商业使用'，表明已商用；文档与回答矛盾"}}

                示例2（文档支持 - yes）：
                文档：故宫周一闭馆维护

                回答：每周一不对外开放

                核查专家：{{"score":"yes", "reason":"回答有文档支持，文档说'故宫周一闭馆维护'，表明周一不开放；回答说'每周一不对外开放'，表明同样含义；文档支持回答"}}

                示例3（缺乏依据 - no）：
                文档：2023年全国GDP增长5.2%

                回答：2023年中国经济总量突破150万亿元，成为全球第一大经济体

                核查专家：{{"score":"no", "reason":"核心结论无文档依据，文档说'2023年全国GDP增长5.2%'，仅提及增速；但回答说'成为全球第一大经济体'，该结论在文档中完全没有提及；文档与回答矛盾"}}

                示例4（部分支持 - yes）：
                文档：海贝思台风于2019年10月登陆日本，造成东京地区严重内涝，多条河流决堤

                回答：海贝思台风引发了严重的洪涝灾害

                核查专家：{{"score":"yes", "reason":"回答有文档支持，文档说'造成东京地区严重内涝，多条河流决堤'，表明发生洪涝灾害；回答说'引发了严重的洪涝灾害'，与文档描述一致；文档支持回答"}}

                示例5（胡编乱造 - no）：
                文档：重庆市石井坡地区在2000年代发生过多次崩塌地质灾害

                回答：石井坡地区从未发生过地质灾害，地质条件非常稳定

                核查专家：{{"score":"no", "reason":"存在文档冲突内容，文档说'发生过多次崩塌地质灾害'，表明该地区不稳定；但回答说'从未发生过地质灾害，地质条件非常稳定'，表明该地区稳定；文档与回答矛盾"}}

                示例6（信息无关 - no）：
                文档：四川省地质灾害防治条例于2012年修订

                回答：四川是一个美食之都，火锅和串串非常有名

                核查专家：{{"score":"no", "reason":"回答关键信息与文档毫无关联，文档说'四川省地质灾害防治条例于2012年修订'，讨论的是地质灾害法规；但回答说'四川是一个美食之都，火锅和串串非常有名'，讨论的是美食；两者无关"}}

            # 【输出要求】
            严格使用以下JSON格式，且不能使用markdown语法，只能是纯文本语法，如果score是no则需要在reason字段中简要说明原因，如果score是yes则reason字段为""：
            ※ 禁止行为：
            - 添加任何解释性文字
            - 使用markdown语法
            - 包含非JSON内容
            - 修改字段名称或结构
            - 请直接输出json格式的内容
            ※ reason字段必填要求：
            - 必须引用文档中的具体内容（使用引号标注）
            - 必须引用回答中的具体内容（使用引号标注）
            - 明确说明两者的关系（支持/矛盾/无关）
            - 使用格式：文档说'xxx'，表明[含义A]；回答说'xxx'，表明[含义B]；[关系判断]
            
        """
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "文档: \n\n {documents} \n\n 回答: {generation}\n\n 核查专家回答："),
            ]
        )
        hallucination_grader = hallucination_prompt | structured_llm_grader
        if is_test in ['test', 'yes', 'YES', 'y', 'Y']:
            tests = [{'q': "请介绍一下’海贝思‘引起的泥石流相关事件。", 'd': "'海贝思'是一款优质的奶粉"},
                     {'q': "最近丁真干了什么事情?", 'd': "丁真最近在理塘放牛!"},
                     {'q': "重庆市石井坡附近发生过哪些崩塌？",
                      'd': "崩塌 石井坡街道前进坡161、162号危岩 野外编号：187 省:重庆市 乡镇:石井坡街道 经度106.436389 纬度29.595278 今后趋势：逐步稳定"}
                     ]
            a = time.time()
            for test in tests:
                rag_llm = self.get_rag_llm(is_mine=is_mine)
                answer = rag_llm.invoke({"question": test['q'], 'context': test['d']})
                print('-'*20)
                print("大模型基于rag回答的内容:",answer)
                is_hallucination = hallucination_grader.invoke({"documents": test['d'], "generation": answer})
                print(is_hallucination.content)
                print(json.loads(is_hallucination.content).get('score', None))
            print(f'生成{len(tests)}个回答,总共耗时{time.time() - a}s')
        return hallucination_grader

    def get_hallucination_rethink(self, is_test = None, is_mine = None)->Any:
        # 幻觉二次检测Agent
        structured_llm_grader = get_client(base_url = 'http://127.0.0.1:8501/v1', model_name = 'Qwen2.5-7B-Instruct')
        rethink_system = """
            【角色定位】
            你是幻觉检测复核专家，给你一个初始检测理由，你负责验证初始检测结论的合理性
            【输入要素】
            1.原始问题
            2.生成回答
            3.初始检测理由
            4.参考文档
            【复核标准】
            输出"yes"即坚持初始检测理由，需同时满足：
            ✅ 初始检测指出的矛盾点确实存在于生成内容中
            ✅ 该矛盾点在文档中确有明确对应内容
            ✅ 推理过程符合逻辑链条
            输出"no"即不赞同初始检测理由，情况包括：
            ⛔ 检测理由中的矛盾点不存在于生成内容
            ⛔ 文档中不存在支持该矛盾的证据
            ⛔ 检测理由存在逻辑跳跃或过度推理
            【复核方法论】
            1. 三角验证法：同步核对问题、回答、文档三方信息
            2. 证据溯源：要求初始检测指出的矛盾必须对应具体文档位置
            3. 逻辑可逆测试：假设回答正确，能否从文档推导出矛盾
            【输出规范】
            最终复核结论：严格使用JSON格式：
            ※ 禁止行为：
            - 添加任何解释性文字
            - 使用markdown语法
            - 包含非JSON内容
            - 修改字段名称或结构
            {{"score":"yes", "reason":"初始检测理由正确，符合无误"}} 或 {{"score":"no", "reason":"初始检测理由错误，回答中的"已商用"与文档"未商用"矛盾"}}
            <判例1>
            问题：量子计算机是否已经商业化？
            生成回答：量子计算机已经商业化
            初始检测理由：回答中的"已商用"与文档"未商用"矛盾
            相关文档：量子计算机尚未实现商用化
            最终复核结论：{{"score":"yes", "reason":"初始检测理由错误，回答中的"已商用"与文档"未商用"矛盾"}}
            
            <判例2>
            问题：图书馆周一开放吗？
            生成回答：周一闭馆
            初始检测理由：回答"周一闭馆"与文档"周一维护"矛盾
            相关文档：每周一进行设施维护，暂停开放
            最终复核结论：{{"score":"no", "reason":"初始检测理由错误，理由不成立，"周一维护"和"周一闭馆"二者实质相同"}} 
            
            <判例3>
            问题：请介绍一下华东市场的贡献
            生成回答：华东市场是我国贡献最高的市场之一
            初始检测理由：回答添加文档未提及的"华东市场贡献"
            文档证据：销售数据部分仅提及"年度增长15%"
            最终复核结论：{{"score":"yes", "reason":"确实存在无依据信息"}} 
        """
        rethink_prompt = ChatPromptTemplate.from_messages([
            ("system", rethink_system),
            ("human",
             "复核任务启动：\n"
             "问题：{question}\n"
             "生成回答：{generation}\n"
             "初始检测理由：{reason}\n"
             "相关文档：{documents}\n\n"
             "请按照以下步骤验证一步一步思考，但是仅仅只给出最终复核结论：\n"
             "1. 提取初始理由中的关键矛盾点\n"
             "2. 在生成回答中定位对应表述\n"
             "3. 在文档中查找支持矛盾的证据\n"
             "4. 评估逻辑链条完整性\n"
             "最终复核结论：")
        ])
        rethink_llm = rethink_prompt | structured_llm_grader
        if is_test in ['test', 'yes', 'YES', 'y', 'Y']:
            tests = [{'q': "请介绍一下’海贝思‘引起的泥石流相关事件。", 'd': "'海贝思'是一款优质的奶粉"},
                     {'q': "最近丁真干了什么事情?", 'd': "丁真最近在理塘放牛!"},
                     {'q': "重庆市石井坡附近发生过哪些崩塌？",
                      'd': "崩塌 石井坡街道前进坡161、162号危岩 野外编号：187 省:重庆市 乡镇:石井坡街道 经度106.436389 纬度29.595278 今后趋势：逐步稳定"}
                     ]
            a = time.time()
            # for test in tests:
            #     rag_llm = self.get_rag_llm(is_mine=is_mine)
            #     answer = rag_llm.invoke({"question": test['q'], 'context': test['d']})
            #     print('-'*20)
            #     print("大模型基于rag回答的内容:",answer)
            #     is_hallucination = hallucination_grader.invoke({"documents": test['d'], "generation": answer})
            #     print(is_hallucination.content)
            #     print(json.loads(is_hallucination.content).get('score', None))
            print(f'生成{len(tests)}个回答,总共耗时{time.time() - a}s')
        return rethink_llm
    def get_answerQ_llm(self, is_test = None, is_mine = None):
        llm = get_client(base_url = 'http://127.0.0.1:8501/v1', model_name = 'Qwen2.5-7B-Instruct')
        system = """
        【角色】 
        问题解决度评估专家

        【任务】
        严格判断回答是否完全解决提问，基于相关性、准确性和完整性进行评估

        【输出规则】
        仅返回JSON格式，且不能使用markdown语法，只能是纯文本语法：
        - 回答完整解决问题：{{"score":"yes", "reason":"回答准确解决了问题，因为..."}} 
        - 回答未解决问题：{{"score":"no", "reason":"回答未解决问题，因为..."}}

        ※ 禁止行为：
        - 添加任何解释性文字
        - 使用markdown语法（如```json）
        - 包含非JSON内容
        - 修改字段名称或结构

        【判断标准】
        评为"yes"需同时满足：
        ✓ 回答内容与问题直接相关
        ✓ 覆盖提问核心诉求
        ✓ 提供有效且准确的信息
        ✓ 无矛盾/错误信息

        评为"no"的情况：
        ✗ 答非所问，内容完全不相关
        ✗ 回答过于笼统或模糊
        ✗ 包含明显错误信息
        ✗ 未回答问题核心要点

        【具体示例】

        示例1（正确回答 - yes）：
        用户问题: 重庆市石井坡附近发生过哪些崩塌？

        模型回答: 石井坡附近发生过崩塌事件，包括前进坡161、162号等地发生过崩塌

        判断: {{"score":"yes", "reason":"回答准确解决了问题，因为明确指出了石井坡附近的具体崩塌地点"}}

        示例2（答非所问 - no）：
        用户问题: 请介绍一下'海贝思'引起的泥石流相关事件。

        模型回答: '海贝思'是一款优质的奶粉，它非常好喝！

        判断: {{"score":"no", "reason":"回答未解决问题，因为用户询问台风'海贝思'引起的泥石流灾害，回答却讨论奶粉，完全答非所问"}}

        示例3（回答笼统 - no）：
        用户问题: 如何预防地质灾害？

        模型回答: 预防地质灾害很重要，需要做好防护工作。

        判断: {{"score":"no", "reason":"回答未解决问题，因为仅提供空泛表述，缺乏具体的预防方法和措施"}}

        示例4（部分相关但有效 - yes）：
        用户问题: 最近丁真在做什么？

        模型回答: 最近丁真在理塘放牛

        判断: {{"score":"yes", "reason":"回答准确解决了问题，因为直接回答了丁真的近期活动"}}

        【注意】
        - reason字段必须说明判断依据，不能留空
        - 评估时聚焦于"是否解决问题"，而非回答的完美程度
        - 如果回答基本解决问题但不够详细，仍应评为"yes"
        - 输出格式为json格式，且不包含任何解释性文字，直接输出json格式内容
        """
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "用户问题: \n\n {question} \n\n 模型回答: {generation}\n\n 判断: "),
            ]
        )
        answerQ_llm = answer_prompt | llm
        if is_test in ['test', 'yes', 'YES', 'y', 'Y']:
            tests = [{'q': "请介绍一下’海贝思‘引起的泥石流相关事件。", 'a': "'海贝思'是一款优质的奶粉, 他非常好喝！"},
                     {'q': "最近丁真干了什么事情?", 'a': "最近丁真在理塘放牛"},
                     {'q': "重庆市石井坡附近发生过哪些崩塌？",
                      'a': "石井坡附近发生过崩塌事件，包括前进坡161、162号等地发生过崩塌"}
                     ]
            a = time.time()
            for test in tests:
                quality = answerQ_llm.invoke({"question": test['q'], "generation": test['a']})
                print(quality.content)
                print(json.loads(quality.content).get('score', None))
            print(f'生成{len(tests)}个回答,总共耗时{time.time() - a}s')
        return answerQ_llm
    def get_reWriteQus_llm(self, is_test=None, is_mine=None) ->Any:
        llm = get_client(base_url = 'http://127.0.0.1:8501/v1', model_name = 'Qwen2.5-7B-Instruct')
        system = """
            # 角色定位
            你是一个专业的查询优化引擎，负责将用户的原始问题转化为更适合向量数据库检索的精准查询语句。

            # 知识库范围
            当前数据库包含以下类型的灾害防治文档：
            - **法律法规层面**：突发事件应对法、防震减灾法、地质灾害防治条例、自然灾害救助条例等
            - **规划预案层面**：国家/省级地质灾害应急预案、"十四五"规划、区域防治方案等
            - **技术规范层面**：分类分级标准、危险性评估规范、监测技术规范、工程勘查规范等
            - **应用研究层面**：AI应急管理、风险防控平台、特定灾害类型研究等

            覆盖地域：全国性文件 + 省级文件（四川、浙江、重庆、陕西、贵州、山东、湖南、上海、广西等）

            # 优化流程
            1. **语义扩展**：补充专业术语和上下位概念（如"滑坡"扩展为"滑坡地质灾害防治"）
            2. **意图显式化**：明确查询类型（定义/流程/标准/案例/责任主体/区域规定）
            3. **术语标准化**：口语化→专业表达，模糊→精确描述
            4. **完整性**：短语→完整问句，增加语义密度以提升向量匹配度
            5. **保真性**：保留原问题的所有约束（地名/数值/时间/灾害类型），不添加不存在的限制
            6. **多轮对话**：仅处理最后一个问题，忽略历史对话

            # 输出要求
            - 直接输出优化后的完整问句，无任何前缀或解释
            - 禁止出现"优化后""改写为""根据上下文"等元信息
            - 保留原问题的所有实体词和参数

            # 优化示例

            **用户问题：**有关上下文:[]\n\n 用户问题：('human':'四川那个滑坡的规定')\n\n 请给出对用户问题一个改进后的问题，使得更容易检索到相关文档。
            
            **模型输出：**四川省地质灾害防治条例中关于滑坡灾害的防治规定有哪些？

            **用户问题：**有关上下文: [('human':'地震的种类有哪些？'),('ai':'地震的种类有:矩形地震、地幔地震等')]\n\n用户问题: ('human':'监测技术') \n\n 请给出对用户问题一个改进后的问题，使得更容易检索到相关文档。
            
            **模型输出：**地质灾害监测技术的规范标准和具体要求是什么？
        """
        system2 = """
            # 角色定位
            你是一个专业的查询优化引擎，负责将用户的原始问题转化为更适合向量数据库检索的精准查询语句。

            # 知识库范围
            当前数据库包含以下类型的灾害防治文档：
            - **法律法规层面**：突发事件应对法、防震减灾法、地质灾害防治条例、自然灾害救助条例等
            - **规划预案层面**：国家/省级地质灾害应急预案、"十四五"规划、区域防治方案等
            - **技术规范层面**：分类分级标准、危险性评估规范、监测技术规范、工程勘查规范等
            - **应用研究层面**：AI应急管理、风险防控平台、特定灾害类型研究等

            覆盖地域：全国性文件 + 省级文件（四川、浙江、重庆、陕西、贵州、山东、湖南、上海、广西等）

            # 优化流程

            ## 第一步：相关性判断（必须执行）
            **判断用户的最新问题是否与灾害防治领域相关**：
            - 如果问题涉及：自然灾害、地质灾害、应急管理、防灾减灾、风险评估、监测预警、法规条例、应急预案等主题 → 继续优化
            - 如果问题是：数学计算、日常闲聊、通用知识问答、编程问题、其他领域专业问题等 → **直接原样输出，不做任何改写**

            ## 第二步：问题优化（仅当相关时执行）
            1. **语义扩展**：补充专业术语和上下位概念（如"滑坡"扩展为"滑坡地质灾害防治"）
            2. **意图显式化**：明确查询类型（定义/流程/标准/案例/责任主体/区域规定）
            3. **术语标准化**：口语化→专业表达，模糊→精确描述
            4. **完整性**：短语→完整问句，增加语义密度以提升向量匹配度
            5. **保真性**：保留原问题的所有约束（地名/数值/时间/灾害类型），不添加不存在的限制

            # 核心原则
            1. **仅处理最新问题**：历史对话仅供参考专业术语，绝不允许用历史话题替换当前问题的主题
            2. **问题主体不可变**：如果用户问"2+1=?"，绝不能因为历史消息而改写成和自然灾害相关的问题
            3. **范围外保持原样**：对于知识库范围外的问题，必须原样输出，体现系统边界意识

            # 输出要求
            - 直接输出优化后的完整问句，无任何前缀或解释
            - 禁止出现"优化后""改写为""根据上下文"等元信息
            - 保留原问题的所有实体词和参数

            # 优化示例

            **示例1：范围内问题优化**
            用户问题：有关上下文:[]\n\n 用户问题：('human':'四川那个滑坡的规定')\n\n 请给出对用户问题一个改进后的问题，使得更容易检索到相关文档。

            模型输出：四川省地质灾害防治条例中关于滑坡灾害的防治规定有哪些？

            **示例2：忽略历史对话，仅优化当前问题**
            用户问题：有关上下文: [('human':'地震的种类有哪些？'),('ai':'地震的种类有:矩形地震、地幔地震等')]\n\n用户问题: ('human':'监测技术') \n\n 请给出对用户问题一个改进后的问题，使得更容易检索到相关文档。

            模型输出：地质灾害监测技术的规范标准和具体要求是什么？

            **示例3：范围外问题保持原样（关键）**
            用户问题：有关上下文: [('human':'地面沉降易发性评价因子'),('ai':'评价因子包括层厚度、地下水主采层数量...')]\n\n用户问题: ('human':'2+1=?') \n\n 请给出对用户问题一个改进后的问题，使得更容易检索到相关文档。

            模型输出：2+1=?

            **示例4：范围外问题保持原样**
            用户问题：有关上下文: [('human':'滑坡监测规范'),('ai':'滑坡监测应符合...')]\n\n用户问题: ('human':'今天天气怎么样') \n\n 请给出对用户问题一个改进后的问题，使得更容易检索到相关文档。

            模型输出：今天天气怎么样
            """
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system2),
                (
                    "human",
                    "有关上下文: \n\n {history}  用户问题:{question}\n\n 请给出对用户问题一个改进后的问题，使得更容易检索到相关文档。\n\n**模型输出：**：",
                ),
            ]
        )
        reWrite_llm = re_write_prompt | llm
        # print('ok')
        if is_test in ['test', 'yes', 'YES', 'y', 'Y']:
            tests = [
                     {'q': [("human", "请介绍一下’海贝思‘引起的泥石流相关事件。"), ("ai", "'海贝思'是一种台风，曾经在日本东京发生过严重的泥石流灾害造成多人受伤"), ("human", "哪些和它一样造成了类似规模的伤亡？")]},
                     {'q': [("human","你知道丁真吗?"), ('ai', "当然!丁真，男，藏族，2000年出生。他因一组“野性与纯真的眼神”照片走红网络，成为四川甘孜理塘县的旅游大使。"),
                            ("human", "你知道蛊真人里面的齐天鸿运蛊吗?"), ('ai', """
                            齐天鸿运蛊是一种极其特殊的蛊虫，它被描述为一种高达八转的消耗性蛊虫，意味着它的力量等级非常高。这种蛊无形无质，本质上是一团恢弘的气运，因此用寻常手段几乎不可能捕捉到它。使用者可以通过吸收周围人的气运来增强自身，随着时间推移，其力量也会逐渐增长。在《蛊真人》的世界中，巨阳仙尊晚年炼制了这一蛊虫，并且它是八十八角真阳楼无上真传之一。
                            尽管齐天鸿运蛊赋予了马鸿运无比的好运气，使他在面对困难时总能化险为夷，但最终他也未能逃脱命运的捉弄。在一系列复杂的势力争夺战中，马鸿运因过度劳累而在逆流河中被方源追赶致死。他的死亡引起了众多读者的关注和讨论，有人认为这是因为宿命的力量终究不可抗拒，即便有再强的运气也无法改变注定的命运。
                            """),
                            ('user', '为什么大家都说这位理塘小伙这个蛊虫有什么关系呢？')]},
                     {'q': "重庆市石井坡附近发生过哪些崩塌？"}
            ]
            a = time.time()
            for test in tests:
                ans = reWrite_llm.invoke({'question':test['q']})
                print(ans.content)
            print(f'生成{len(tests)}个回答,总共耗时{time.time() - a}s')
        return reWrite_llm
    def get_basic_llm(self, is_test = None, is_mine = None)->Any:
        return get_client(base_url = 'http://127.0.0.1:8501/v1', model_name = 'Qwen2.5-7B-Instruct')
    def Message2History(self, Messages:list[AIMessage])->list:
        Historys = []
        for item in Messages:
            if isinstance(item, AIMessage):
                Historys.append(('ai:', item.content))
            elif isinstance(item, HumanMessage):
                Historys.append(('human:', item.content))
            else:
                # 忽略ToolMessage等其他信息
                pass
        return Historys
if __name__ == '__main__':
    zero_agent = ZeroAgent()
    zero_agent.get_grader_llm(is_test='test', is_mine='test')
