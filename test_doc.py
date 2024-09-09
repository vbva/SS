import os
import glob
from docx import Document
import pdfplumber
from datasets import Dataset
import chromadb
from model import preprocess

# Функция для извлечения текста из DOCX
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Функция для извлечения текста из PDF
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Загрузка документов из папки
def load_documents_from_folder(folder_path):
    documents = []
    # Загрузка DOCX файлов
    for file_path in glob.glob(os.path.join(folder_path, '*.docx')):
        text = extract_text_from_docx(file_path)
        documents.append({"file_name": os.path.basename(file_path), "text": text})
    
    # Загрузка PDF файлов
    for file_path in glob.glob(os.path.join(folder_path, '*.pdf')):
        text = extract_text_from_pdf(file_path)
        documents.append({"file_name": os.path.basename(file_path), "text": text})
    
    return documents

# Путь к директории с документами DOCX и PDF
folder_path = "./data"
documents = load_documents_from_folder(folder_path)
documents = documents[:80]  # Ограничение в 100 документов

# Создаем локальное хранилище ChromaDB
client = chromadb.PersistentClient(path="./database_80_test")

# Создание коллекции, если она еще не создана
collection = client.get_or_create_collection(
    name="CB_collection_test",
    metadata={"hnsw:space": "cosine"}
)

# Параметр для размера батча
batch_size = 10

# Обрабатываем документы порциями по batch_size документов
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    
    # Создание Dataset из батча документов
    ds = Dataset.from_dict({"file_name": [doc["file_name"] for doc in batch],
                            "text": [doc["text"] for doc in batch]})
    
    # Применение функции preprocess к текстам для получения эмбеддингов
    ds_embs = ds.map(preprocess, batched=True)
    
    # Добавление эмбеддингов в коллекцию ChromaDB
    collection.add(
        documents=ds_embs["text"],        # Тексты документов
        embeddings=ds_embs["embeddings"], # Эмбеддинги, сгенерированные функцией preprocess
        ids=[f"id{i+j+1}" for j in range(len(ds_embs))]  # Уникальные идентификаторы документов
    )
    
    # Выводим информацию о текущем статусе
    print(f"Добавлено {len(ds_embs)} документов в коллекцию (batch {i // batch_size + 1}).")

# Выводим информацию о завершении процесса
print(f"Добавление всех документов завершено. Всего добавлено {len(documents)} документов.")
