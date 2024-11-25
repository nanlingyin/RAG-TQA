import os
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

client = OpenAI(
    api_key="your_api_key",
    base_url="https://api.ephone.chat/v1"
)



def load_triviaqa(filepath):
    dataset = load_dataset("parquet", data_files=filepath)
    texts = []
    for idx, example in enumerate(dataset["train"]):  
        context = example.get("context", "").strip()
        texts.append(context)
    return texts

def initialize_retriever(texts):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(texts)
    return vectorizer, vectors

def retrieve(query, vectorizer, vectors, texts, top_k=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    retrieved_texts = [texts[i] for i in top_indices]
    return " ".join(retrieved_texts)

def generate_answer(question, context):
    prompt = f"根据以下上下文回答问题：\n\n上下文: {context}\n\n问题: {question}\n答案:"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
        ]
    )
    answer = response.choices[0].message.content.strip()
    return answer

def main():
    dataset_path = r'C:\Users\admin\Desktop\RAG-TQA\TriviaQA\modified_100_percent-00000-of-00001-6ecbad160e20a7c4.parquet'
    texts = load_triviaqa(dataset_path)

    
    vectorizer, vectors = initialize_retriever(texts)
    
    while True:
        question = input("请输入您的问题 (或输入'退出'结束): ")
        if question.lower() == '退出':
            break
        context = retrieve(question, vectorizer, vectors, texts)
        answer = generate_answer(question, context)
        print(f"回答: {answer}\n")

if __name__ == "__main__":
    main()