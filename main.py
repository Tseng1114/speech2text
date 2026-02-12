import os
from google import genai
from google.genai import types
import semantic_similarity  

########################################################################################
# --------------------------  Gemini API Tokens estimate   --------------------------- #  
#                   Monthly limit: 250,000(250K) TPM (Tokens Per Month)                #
#                   Example: A 4 min 30 sec video consumes ~8,000(8K) tokens           # 
########################################################################################

client = genai.Client(api_key="API KEY")             # replace with your actual API key

base_dir = os.path.dirname(__file__)
audio_folder = os.path.join(base_dir, "audio_sample")#
transcript_folder = os.path.join(base_dir, "output")  
os.makedirs(transcript_folder, exist_ok=True)
ground_truth_path = os.path.join(base_dir, "test_model", "ground_truth.txt")

for filename in os.listdir(audio_folder):
    if not filename.lower().endswith((".mp3", ".wav", ".m4a")):
        continue  

    audio_path = os.path.join(audio_folder, filename)#
    transcript_file = os.path.join(transcript_folder, f"{os.path.splitext(filename)[0]}.txt")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    response = client.models.generate_content(
        model="gemini-2.5-flash",                    # change model if needed
        contents=[
            "Generate a transcript of the speech.",  
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type="audio/mp3"                # adjust MIME type based on your audio format      
            )
        ],
    )
    transcript_text = response.text

    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    print(f"[{filename}] transcription saved to {transcript_file}.")
    
###############################     test model section    ##############################
#####      this section compares the transcript with the ground truth text         ##### 
#   similarity_percent = semantic_similarity.compare(transcript_file, ground_truth_path) 
#   print(f"[{filename}] Semantic Similarity: {similarity_percent:.2f}%\n")
##### If you want to test the model, just remove the "#" from the two lines above.  ####
###############################     test model section    ##############################