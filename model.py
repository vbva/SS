from sentence_transformers import SentenceTransformer, CrossEncoder, util
#import chromadb
#from make_ds_from_docs import ds 
from datasets import load_dataset

CACHE_DIR = "./hf_cache_3"

retriever = SentenceTransformer(
    model_name_or_path="DiTy/bi-encoder-russian-msmarco",
    cache_folder=CACHE_DIR,
)

reranker_model = CrossEncoder(
    model_name="DiTy/cross-encoder-russian-msmarco",
    max_length=512,     
    automodel_args={
        "cache_dir": CACHE_DIR,
    },
    tokenizer_args={
        "cache_dir": CACHE_DIR,
    }
)

def preprocess(example):
    if "embeddings" not in example:
        example["embeddings"] = retriever.encode(example["text"], convert_to_tensor=True)
    return example

