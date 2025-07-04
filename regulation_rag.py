import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


"""
2. 将文档加载到向量数据库 chroma 中
"""

from model import RagEmbedding
embedding_cls = RagEmbedding()

from langchain_chroma import Chroma
import chromadb
chroma_client = chromadb.PersistentClient(path='./chroma')

# 检查 chroma 目录是否存在且有内容
chroma_has_data = os.path.exists('./chroma') and len(os.listdir('./chroma')) > 0

# 如果 chroma 目录不存在或为空，创建新的数据库
if not chroma_has_data:
    from doc_parse import chunk, read_and_process_excel
    import pandas as pd

    """
    1. 解析文档
    """

    pdf_files = ['./data/zhidu_employee.pdf', './data/zhidu_travel.pdf']
    excel_files = ['./data/zhidu_detail.xlsx']

    doc_data = []
    for pdf_file_name in pdf_files:
        res = chunk(pdf_file_name)
        doc_data.extend(res)

    print(f"解析文档个数：{len(doc_data)}")
    for doc in doc_data:
        print("#"*20)
        print(doc.page_content)

    from langchain_core.documents import Document

    for excel_file_name in excel_files:
        data = read_and_process_excel(excel_file_name)
        df = pd.DataFrame(data[8:], columns=data[7])
        data_excel = df.drop(columns=df.columns[11:17])
        doc_excel = Document(
            page_content = data_excel.to_markdown(index=False).replace(" ", ""),
            metadata = {"source": excel_file_name}
        )
        print(f"doc excel: {doc_excel}")
        doc_data.append(doc_excel)

    print(f"解析文档个数：{len(doc_data)}")

    zhidu_db = Chroma.from_documents(doc_data,
                            embedding_cls.get_embedding_fun(),
                            client=chroma_client,
                            collection_name="zhidu_db")
    print("创建新的向量数据库")
else:
    # 如果 chroma 目录存在且有内容，直接加载已有的数据库
    zhidu_db = Chroma(
        collection_name="zhidu_db",
        embedding_function=embedding_cls.get_embedding_fun(),
        client=chroma_client
    )
    print("从 chroma 文件中加载向量文档")

"""
3. 构建提示词
"""

prompt_template = """
你是企业员工助手，熟悉公司考勤和报销标准等规章制度，需要根据提供的上下文信息 context 来回答员工的提问。\
请直接回答问题，如果上下文信息 context 中没有和问题相关的信息，请直接回答[不知道，请咨询 HR] \
问题：{question}
"{context}"
回答：
"""

"""
4. 构建 RAG pipeline
"""

from langchain import PromptTemplate
from model import RagLLM
llm = RagLLM()

def run_rag_pipeline(query, context_query, k=3, context_query_type="query",
                     stream=True, prompt_template=prompt_template,
                     temperature=0.1):
    if context_query_type == "vector":
        related_docs = zhidu_db.similarity_search_by_vector(context_query, k=k)
    elif context_query_type == "query":
        related_docs = zhidu_db.similarity_search(context_query, k=k)
    elif context_query_type == "doc":
        related_docs = context_query
    else:
        related_docs = zhidu_db.similarity_search(context_query, k=k)
    context = "\n".join([f"上下文{i+1}: {doc.page_content} \n" \
                         for i, doc in enumerate(related_docs)])
    print()
    print()
    print("#"*100)
    print(f"query: {query}")
    print(f"context: {context}")
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=prompt_template,
    )
    llm_prompt = prompt.format(question=query, context=context)

    if stream:
        response = llm.stream(llm_prompt)
        print("stream response:")
        for chunk in response:
            print(chunk, end='', flush=True)
        return ""
    else:
        response = llm(llm_prompt, stream=True, temperature=temperature)
        print(f"response: \n{response}")
        return response