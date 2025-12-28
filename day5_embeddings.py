from google import genai

client = genai.Client(api_key="AIzaSyAI_PdMHZTxWlWjV03UC2R-atML6AUWdwI")

#Task-1
response = client.models.embed_content(
    model="text-embedding-004",
    contents="Artificial Intelligence is shaping the future"
)

print(response)

#Task-2
import numpy as np

def cosine_similarity(vec1, vec2):
  return np.dot(vec1,vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))

text1 = "AI is transforming industries"
text2 = "Artificial intelligence is changing businesses"

emb1 = client.models.embed_content(
    model="text-embedding-004",
    contents=text1
)

emb2 = client.models.embed_content(
    model="text-embedding-004",
    contents=text2
)

emb1 = np.array(emb1.embeddings[0].values)
emb2 = np.array(emb2.embeddings[0].values)

similarity = cosine_similarity(emb1, emb2)
print("Similarity:", similarity)

#Task-3
text1 = "I love machine learning"
text2 = "Deep learning models are powerful"
text3 = "The weather is very hot today"

emb1 = client.models.embed_content(
    model="text-embedding-004",
    contents=text1
)

emb2 = client.models.embed_content(
    model="text-embedding-004",
    contents=text2
)

emb3 = client.models.embed_content(
    model="text-embedding-004",
    contents=text3
)

emb1 = np.array(emb1.embeddings[0].values)
emb2 = np.array(emb2.embeddings[0].values)
emb3 = np.array(emb3.embeddings[0].values)

print("Text1 vs Text2:", cosine_similarity(emb1, emb2))
print("Text1 vs Text3:", cosine_similarity(emb1, emb3))

#Task-4
import numpy as np

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

sentences = [
      "I enjoy studying AI",
      "Football is a popular sport",
      "Machine learning models learn from data",
      "The sky is very blue today"
]

query = "learning artificial intelligence"

query_response = client.models.embed_content(
    model="text-embedding-004",
    contents=query
)
query_emb = np.array(query_response.embeddings[0].values)

scores = []

for sent in sentences:
    sent_response = client.models.embed_content(
        model="text-embedding-004",
        contents=sent
    )
    sent_emb = np.array(sent_response.embeddings[0].values)

    score = cosine_similarity(query_emb, sent_emb)
    scores.append((sent, score))

scores.sort(key=lambda x: x[1], reverse=True)

for s, sc in scores:
    print(s, "->", sc)
