import os
import re
from sentence_transformers import SentenceTransformer, util

def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    text = "".join(text.split())
    return text

def compare(transcript_path, ground_truth_path):
    if not os.path.exists(ground_truth_path):
        return None

    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read().strip()
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            ground_truth_text = f.read().strip()
    except Exception as e:
        print(f"Read file error: {e}")
        return 0

    t_clean = clean_text(transcript_text)
    g_clean = clean_text(ground_truth_text)

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    t_vec = model.encode(t_clean, convert_to_tensor=True)
    g_vec = model.encode(g_clean, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(t_vec, g_vec).item()
    similarity_percent = similarity * 100

    return similarity_percent