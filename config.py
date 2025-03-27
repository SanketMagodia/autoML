import os
from dotenv import load_dotenv
# Configuration for the language model
#install appropriate AutoGen lib for your desired llm Api

load_dotenv()
config_list = [
    {
        "model": "llama-3.3-70b-specdec", 
        "api_key": os.environ.get("GROQ_API_KEY"), 
        "api_type": "groq"
    }
]