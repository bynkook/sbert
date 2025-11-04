# pip install -U sentence-transformers huggingface_hub

from sentence_transformers import SentenceTransformer

# 1) snunlp/KR-SBERT-V40K-klueNLI-augSTS
m1 = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
m1.save("./kr-sbert-v40k-kluenli-augsts")  # 로컬 경로

# 2) BM-K/KoSimCSE-roberta
# m2 = SentenceTransformer("BM-K/KoSimCSE-roberta")
# m2.save("./models/kosimcse-roberta")

# 3) upskyy/e5-large-korean
# m3 = SentenceTransformer("upskyy/e5-large-korean")
# m3.save("./models/e5-large-korean")

# 4) upskyy/gte-base-korean  (모델 카드 권고에 따라 trust_remote_code 필요할 수 있음)
# m4 = SentenceTransformer("upskyy/gte-base-korean", trust_remote_code=True)
# m4.save("./models/gte-base-korean")