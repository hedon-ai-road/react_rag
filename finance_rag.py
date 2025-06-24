from model import RagLLM
from langchain import PromptTemplate

"""
1. 使用 LLM 提取问题中的关键字
"""

kg_llm = RagLLM()

prompt = """给定一些初始查询，提取最多 {max_keywords} 个相关的关键词，
不一定是原文里面的词，可以是其相关的近义词或是抽象概念，
但是要考虑大小写、复数形式、常见表达等。
用 '^' 符号分割所有同义词 / 关键词：'关键词1^关键词2^...'

注意，结果应为一行，用 '^' 符号分隔。
---
查询：{query_str}
---
关键词:
"""

prompt_template = PromptTemplate(
    input_variables=["query_str", "max_keywords"],
    template=prompt,
)

def parse_query(query, max_keywords=3):
    prompt = prompt_template.format(max_keywords=max_keywords,
                                    query_str=query)
    resp = kg_llm(prompt)
    keywords = resp.split('\n')[0].split('^')
    return keywords

"""
2. 通过关键字尝试在图数据库中找到相关的节点和关系
"""

from py2neo import Graph, Node, Relationship
graph = Graph("bolt://localhost:7687", user='neo4j', password='neo4j123', name='neo4j')

def get_node(keyword, node_type):
    query = f"""
    MATCH (n:{node_type})
    where n.name CONTAINS "{keyword}"
    RETURN n.name as name
    """

    results = graph.run(query)
    print(query)
    for record in results:
        return record['name']
    return None

def gen_contexts(investor_condition,
                 company_condition,
                 event_type_condition,
                 query_level=1,
                 exclude_content=False):
    if query_level == 1:
        query = f"""
        MATCH (i:Investor)-[:INVEST]->(c:Company)-[r:HAPPEN]->(e)
        WHERE 1=1 {investor_condition} {company_condition} {event_type_condition}
        RETURN i.name as investor, c.name as company_name, e.name as event_type, r as relation
        """
    else:
        query = f"""
        MATCH (c:Company)-[r:HAPPEN]->(e)
        WHERE 1=1 {company_condition} {event_type_condition}
        RETURN c.name as company_name, e.name as event_type, r as relation
        """
    print(query)
    results = graph.run(query)
    contexts = []
    for record in results:
        context = ''
        record = dict(record)
        if 'investor' in record:
            context += f"{record['investor']} 投资了 {record['company_name']} \n"
        context = context + f"{record['company_name']} 发生了 {record['event_type']} \n 详细如下："
        for key, value in dict(record['relation']).items():
            if exclude_content:
                if key in ["title", "content"]:
                    continue
            context = context + f"\n  {key}: {value}"
        contexts.append(context)
    return contexts

def get_event_detail(keyword, exclude_content=False):
    investor = get_node(keyword, "Investor")
    company = get_node(keyword, "Company")
    event_type = get_node(keyword, "EventType")

    investor_condition = ""
    company_condition = ""
    event_type_condition = ""
    if investor:
        investor_condition = f' and i.name = "{investor}"'
    if company:
        company_condition = f' and c.name = "{company}"'
    if event_type:
        event_type_condition = f' and e.name = "{event_type}"'

    print(f"investor={investor_condition} company={company_condition} event_type={event_type_condition}")
    if investor_condition or company_condition or event_type_condition:
        contexts = gen_contexts(investor_condition,
                                company_condition,
                                event_type_condition,
                                query_level=1,
                                exclude_content=exclude_content)
        if len(contexts) == 0:
            contexts = gen_contexts(investor_condition,
                                company_condition,
                                event_type_condition,
                                query_level=2,
                                exclude_content=exclude_content)
        return contexts
    else:
        return []

"""
3. 执行 graph rag pipeline
"""

rag_llm = RagLLM()

finance_template = """
你是金融知识助手，熟悉各种金融事件，需要根据提供的上下文信息 context 来回答员工的提问。\
请直接回答问题，如果上下文信息 context 没有和问题相关的信息，请直接先回答不知道,要求用中文输出 \
问题：{question}
上下文信息：
"{context}"
回答：
"""

def graph_rag_pipeline(query, exclude_content=True, stream=True, temperature=0.1):
    keywords = parse_query(query=query, max_keywords=3)
    contexts = []
    ignore_words = ['公司', '分析', '投资']
    for keyword in keywords:
        if keyword in ignore_words:
            continue
        contexts.extend(get_event_detail(keyword=keyword, exclude_content=exclude_content))
    
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=finance_template,
    )
    llm_prompt = prompt.format(question=query, context="\n========================\n".join(contexts))
    print(llm_prompt)

    if stream:
        response = rag_llm.stream(llm_prompt)
        print("response:")
        for chunk in response:
            print(chunk, end='', flush=True)
        return ""
    else:
        response = rag_llm(llm_prompt, temperature=temperature)
        return response