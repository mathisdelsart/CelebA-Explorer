import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(embeddings, index, image_urls, method='cosine'):
    similarities = []

    if isinstance(image_urls, pd.Series):
        image_urls = image_urls.tolist()
    
    if index < 0 or index >= len(embeddings):
        raise ValueError(f"L'index {index} est hors des limites des embeddings.")

    embedding = embeddings.iloc[index].values.reshape(1, -1)
    
    for index2 in range(len(embeddings)):
        if index2 == index:
            continue
        
        embedding2 = embeddings.iloc[index2].values.reshape(1, -1)
        
        if method == 'cosine':
            similarity_score = cosine_similarity(embedding, embedding2)[0][0]
        elif method == 'euclidean':
            similarity_score = 1 / (1 + np.linalg.norm(embedding - embedding2))
        else:
            raise ValueError(f"Le mode de similarité '{method}' n'est pas reconnu.")
        
        similarities.append((image_urls[index2], index2, similarity_score))
    
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities