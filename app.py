__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from model import retriever, reranker_model
import chromadb

# Подключаемся к существующей базе данных
client = chromadb.PersistentClient(path="./database_80_test")
collection = client.get_collection("CB_collection_test")

# Функция для подготовки запроса
def prepare_query(query: str) -> list:
    return retriever.encode(query).tolist()

# Основная функция для поиска и ранжирования документов
def retrieve_and_rerank(query: str, top_k=3, return_only_top1=False):
    query_embeddings = prepare_query(query)
    
    # Выполняем запрос к коллекции
    retrieve_results = collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k*12,
        include=['documents']
    )

    # Проверка, что запрос вернул результаты
    if len(retrieve_results['documents']) == 0 or len(retrieve_results['documents'][0]) == 0:
        raise ValueError("Коллекция не вернула ни одного документа")

    # Re-Rank top-k документов
    rerank_results = reranker_model.rank(
        query=query,
        documents=retrieve_results['documents'][0],
        top_k=top_k
    )

    # Проверка, что ранжировщик вернул результаты
    if len(rerank_results) == 0:
        raise ValueError("Ранжировщик не смог обработать документы")

    # Возвращаем ранжированные результаты
    if return_only_top1:
        top_result = rerank_results[0]
        return {
            'rank': 1,
            'candidate': retrieve_results['documents'][0][top_result['corpus_id']],
            'score': top_result['score']
        }
    else:
        # Вывод всех ранжированных результатов
        return [
            {
                'rank': i + 1,  # Порядковый номер (ранг)
                'candidate': retrieve_results['documents'][0][rerank_results[i]['corpus_id']],
                'score': rerank_results[i]['score']
            }
            for i in range(len(rerank_results))
        ]

# Интерфейс Streamlit
st.title("Поиск и ранжирование документов")

query = st.text_input("Введите запрос:", "")

if st.button("Поиск"):
    if query:
        try:
            results = retrieve_and_rerank(query, top_k=5)
            for res in results:
                st.markdown(f"**Ранг:** {res['rank']}")
                st.markdown(f"**Оценка:** {res['score']:.4f}")
                st.markdown("**Документ:**")
                st.write(res['candidate'])
                st.markdown("---")  # Разделитель между результатами
        except ValueError as e:
            st.error(f"Ошибка: {e}")
    else:
        st.warning("Пожалуйста, введите запрос.")
