from scipy.spatial.distance import cosine
from src.embedder import Embedder

class FilterSimilar:

    def __init__(self, threshold : float = 0.6) -> None:
        self.embedder = Embedder()
        self.threshold = threshold

    def embed_keywords(self,keywords: list) -> dict:
        vecotrs = self.embedder(keywords)
        return { k : v for k , v in zip(keywords, vecotrs)}
    

    def __call__(self, keywords_1 : list, keywords_2 : list) -> list:


        vec_1_mapping = self.embed_keywords(keywords_1)
        vec_2_mapping = self.embed_keywords(keywords_2)

        unique_keywords = []

        unique_keywords.extend(list(vec_1_mapping.keys()))

        for k_2, v_2 in vec_2_mapping.items():
            is_unique = True
            for k_1, v_1 in vec_1_mapping.items(): 

                similarity = 1 - cosine(v_1,v_2)

                if similarity > self.threshold:
                    is_unique = False
                    break
            
            if is_unique:
                unique_keywords.append(k_2)

        return unique_keywords