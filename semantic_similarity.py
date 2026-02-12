import os
import re
from sentence_transformers import SentenceTransformer, util

def clean_text(text: str) -> str:

    text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)      # delete all characters that are not English, Chinese, underscores, numbers or underscores.
    text = "".join(text.split())
    return text

def compare(transcript_path: str, ground_truth_path: str) -> float:
 
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read().strip()
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            ground_truth_text = f.read().strip()
    except Exception as e:
        print(f"read file error: {e}")
        return 0.0

    t_clean = clean_text(transcript_text)
    g_clean = clean_text(ground_truth_text)

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    t_vec = model.encode(t_clean, convert_to_tensor=True)
    g_vec = model.encode(g_clean, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(t_vec, g_vec).item()
    similarity_percent = similarity * 100

    return similarity_percent

def compare_folder(transcript_folder: str, ground_truth_path: str):
    if not os.path.exists(transcript_folder):
        print(f"file does't exist in: {transcript_folder}")
        return

    for filename in os.listdir(transcript_folder):
        if filename.lower().endswith(".txt"):
            transcript_file = os.path.join(transcript_folder, filename)
            compare(transcript_file, ground_truth_path)

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    transcript_folder = os.path.join(base_dir, "output")
    ground_truth_file = os.path.join(base_dir, "test_model", "ground_truth.txt")
    compare_folder(transcript_folder, ground_truth_file)
