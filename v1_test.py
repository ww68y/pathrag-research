import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import openai_complete
from dotenv import load_dotenv  # import load_dotenv

# Load environment variables from a .env file in the current directory.
load_dotenv()
WORKING_DIR = os.path.join(os.getcwd(), 'data')

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=openai_complete,  
)

data_file=""
question="What is iTravel?"
# Iterate over all .txt files in the working directory and insert their content
# for filename in os.listdir(WORKING_DIR):
#     if filename.endswith('.txt'):
#         filepath = os.path.join(WORKING_DIR, filename)
#         with open(filepath, 'r', encoding='utf-8') as f:
#             content = f.read()
#             rag.insert(content)
#             print(f"Inserted content from {filename}")

print(rag.query(question, param=QueryParam(mode="hybrid")))














