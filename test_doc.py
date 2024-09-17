import os
import glob
from datasets import Dataset
import chromadb
from model import preprocess

# Функция для извлечения текста из TXT
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Генератор для загрузки документов из папки
def load_documents_from_folder(folder_path):
    # Загрузка TXT файлов
    for file_path in glob.glob(os.path.join(folder_path, '*.txt')):
        text = extract_text_from_txt(file_path)
        yield {"file_name": os.path.basename(file_path), "text": text}

# Функция для записи имен файлов в txt
def save_document_names(doc_names, file_name="doc_names.txt"):
    with open(file_name, 'a') as f:
        for name in doc_names:
            f.write(f"{name}\n")

# Путь к директории с документами TXT
folder_path = "C:/Users/vbato/Desktop/local-LLM-with-RAG/new_txt_data"

documents_gen = load_documents_from_folder(folder_path)  # Генератор документов
batch_size = 20  # Размер батча, можно регулировать

# Создаем локальное хранилище ChromaDB
client = chromadb.PersistentClient(path="./database")

# Создание коллекции, если она еще не создана
collection = client.get_or_create_collection(
    name="CB_collection_test",
    metadata={"hnsw:space": "cosine"}
)

# Инициализация списка для накопления батчей документов
batch_documents = []
document_names = []

# Обрабатываем все документы
for i, document in enumerate(documents_gen):
    batch_documents.append(document)
    document_names.append(document["file_name"])

    # Если накопили batch_size документов
    if (i + 1) % batch_size == 0:
        # Создание Dataset из батча документов
        ds = Dataset.from_dict({"file_name": [doc["file_name"] for doc in batch_documents],
                                "text": [doc["text"] for doc in batch_documents]})

        # Применение функции preprocess к текстам для получения эмбеддингов
        ds_embs = ds.map(preprocess, batched=True)

        # Добавление эмбеддингов в коллекцию ChromaDB
        collection.add(
            documents=ds_embs["text"],        # Тексты документов
            embeddings=ds_embs["embeddings"], # Эмбеддинги, сгенерированные функцией preprocess
            ids=[f"id{i * batch_size + j + 1}" for j in range(len(ds_embs))]  # Уникальные идентификаторы документов
        )
        
        # Сохраняем имена файлов в txt
        save_document_names([doc["file_name"] for doc in batch_documents])

        # Очищаем batch_documents для следующей порции
        batch_documents = []

        # Выводим информацию о текущем статусе
        print(f"Добавлено {len(ds_embs)} документов в коллекцию (batch {i // batch_size + 1}).")

# Добавляем оставшиеся документы, если они не кратны batch_size
if batch_documents:
    ds = Dataset.from_dict({"file_name": [doc["file_name"] for doc in batch_documents],
                            "text": [doc["text"] for doc in batch_documents]})
    
    ds_embs = ds.map(preprocess, batched=True)

    collection.add(
        documents=ds_embs["text"],        
        embeddings=ds_embs["embeddings"], 
        ids=[f"id{i * batch_size + j + 1}" for j in range(len(ds_embs))]
    )

    # Сохраняем имена файлов в txt
    save_document_names([doc["file_name"] for doc in batch_documents])

    print(f"Добавлено {len(ds_embs)} документов в коллекцию (финальный batch).")

# Выводим информацию о завершении процесса
print(f"Добавление всех документов завершено. Всего добавлено {i + 1} документов.")
