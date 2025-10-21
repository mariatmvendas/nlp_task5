import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query, Path, HTTPException
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Load data

with open("courses.json", "r") as f:
    data = json.load(f)

print(len(data), "files loaded.")



# Prepare first corpus as a list of courses
# each course is a string with its title + learnig objectives, all separated by \n
# course id is removed from title

courses_ids=list(data.keys())
courses_corpus=[]
for value in data.values():
    title=value["title"]
    lobj=value["learning-objectives"]
    title_clean=re.sub(r"\d+", "", title).strip()
    text=title_clean + "\n" + "\n".join(lobj)
    courses_corpus.append(text)

no_courses=len(courses_ids)

print(no_courses, "documents in courses corpus")



# Prepare second corpus as a list of course + objective pairs
# one row per objective per course (each objective appears multiple times)
# value corresponds to course title + objective text

obj_corpus=[]
obj_ids=[]
for course_id, value in data.items():
    title=value["title"]
    title_clean=re.sub(r"\d+", "", title).strip()
    for idx_obj, objective in enumerate(value["learning-objectives"]):
        text=title_clean + "\n" + objective
        obj_corpus.append(text)
        obj_ids.append((course_id, idx_obj))

no_obj=len(obj_ids)

print(no_obj, "documents in objectives corpus")



# TF IDF sparse matrix
# lowercase (default), remove stopwords and allow bigrams

tfidf_courses=TfidfVectorizer(stop_words='english',ngram_range=(1, 2))  
sparse_courses=tfidf_courses.fit_transform(courses_corpus)

tfidf_obj=TfidfVectorizer(stop_words='english',ngram_range=(1, 2))  
sparse_obj=tfidf_obj.fit_transform(obj_corpus)

print("Sparse matrices computed.")
print("Sparse matrix dimensions for courses:", sparse_courses.shape)
print("Sparse matrix dimensions for objectives:", sparse_obj.shape)


# Dense embeddings
# normalize embeddings for easy cossine similarity computation

model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
embed_courses = model.encode(courses_corpus, normalize_embeddings=True)
embed_obj = model.encode(obj_corpus, normalize_embeddings=True)

print("Dense embeddings computed.")
print("Dense embeddings dimensions for courses:", embed_courses.shape)
print("Dense embeddings dimensions for objectives:", embed_obj.shape)


# WEB SERVICE

app=FastAPI()



@app.get("/v1/courses/{course_id}/similar")
def similar_courses(
    course_id: str = Path(...),
    top_k: int = Query(10),
    mode: str = Query("dense"),
    ):

    """
    Returns the top_k courses most similar to a given course_id.
    Choose between a sparse (TF-IDF) or dense (embedding) retriever.
    Compare the embeddings of title + learning objectives between the given course and all the possible courses.
    """

    if course_id not in courses_ids:
        raise HTTPException(status_code=404, detail="course ID not found")
    
    max_k=no_courses-1
    if top_k > max_k:
        raise HTTPException(status_code=400,detail=f"top_k must be smaller or equal to {max_k}")
    
    if mode not in ["sparse", "dense"]:
        raise HTTPException(status_code=400, detail="mode must be 'sparse' or 'dense'")
    
    idx=courses_ids.index(course_id)

    if mode=="sparse":
        sim = cosine_similarity(sparse_courses[idx], sparse_courses).flatten()
    else:
        sim = embed_courses[idx] @ embed_courses.T

    top_k_idx = np.argsort(-sim)[1:top_k+1]

    results = [
        {"course_id": courses_ids[i], "title": data[courses_ids[i]]["title"], "score": float(sim[i])}
        for i in top_k_idx
    ]

    return {"query_course_id": course_id,
            "results": results,
            "mode": mode,
            "top_k": top_k}




@app.get("/v1/search")
def free_query(
        query: str = Query("machine learning"),
        top_k: int = Query(10),
        mode: str = Query("dense")
    ):
    
    """
    Returns the top_k courses most similar to the given query.
    Choose between a sparse (TF-IDF) or dense (embedding) retriever.
    Compare the embeddings of title + learning objectives between the given query and all the possible courses.
    """

    if top_k > no_courses:
        raise HTTPException(status_code=400,detail=f"top_k must be smaller or equal to {no_courses}")
    
    if mode not in ["sparse", "dense"]:
        raise HTTPException(status_code=400, detail="mode must be 'sparse' or 'dense'")

    if mode == "sparse":
        query_sparse=tfidf_courses.transform([query])
        sim = cosine_similarity(query_sparse, sparse_courses).flatten()
    else:
        query_embeddings = model.encode(query, normalize_embeddings=True)
        sim = query_embeddings @ embed_courses.T

    top_k_idx = np.argsort(-sim)[:top_k]

    results = [
        {"course_id": courses_ids[i], "title": data[courses_ids[i]]["title"], "score": float(sim[i])}
        for i in top_k_idx
    ]

    return {"query": query,
            "results": results,
            "mode": mode}



@app.get("/v1/objectives/search")
def objectives(
        query: str = Query("dimensionality reduction"),
        top_k: int = Query(10),
        mode: str = Query("dense")
):
    
    """
    Returns the top_k objectives most similar to the given query.
    Choose between a sparse (TF-IDF) or dense (embedding) retriever.
    Compare the embeddings of title + learning objective between the given query and all the possible objectives.
    """

    if top_k > no_obj:
        raise HTTPException(status_code=400,detail=f"top_k must be smaller or equal to {no_obj}")
    
    if mode not in ["sparse", "dense"]:
        raise HTTPException(status_code=400, detail="mode must be 'sparse' or 'dense'")

    if mode == "sparse":
        query_sparse=tfidf_obj.transform([query])
        sim = cosine_similarity(query_sparse, sparse_obj).flatten()
    else:
        query_embeddings = model.encode(query, normalize_embeddings=True)
        sim = query_embeddings @ embed_obj.T

    # top indices of objs
    top_idx = np.argsort(-sim)[:top_k]
    top_id_obj = [obj_ids[i] for i in top_idx]


    results = [
        {"course_id": c_id, 
         "title": data[c_id]["title"], 
         "objective": data[c_id]["learning-objectives"][o_idx],
         "score": float(sim[i])}
        for i, (c_id, o_idx) in enumerate(top_id_obj)
    ]

    return {"query": query,
            "results": results}
    


@app.get("/v1/health")
def health():
    return {"staus": "ok", "index_sizes": {"courses": no_courses, "objectives": no_obj}}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


