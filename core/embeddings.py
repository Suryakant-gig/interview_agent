import os
import torch

from dotenv import load_dotenv

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

# ---------------------------------
# Load env
# ---------------------------------
load_dotenv()

# ---------------------------------
# FORCE OFFLINE MODE
# ---------------------------------
os.environ["HF_HUB_OFFLINE"] = "1"

# ---------------------------------
# Check CUDA
# ---------------------------------
print(
    "CUDA Available:",
    torch.cuda.is_available()
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    DEVICE = "cuda"

else:

    print("Using CPU")

    DEVICE = "cpu"

# ---------------------------------
# HF Token
# ---------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")

# ---------------------------------
# Embedding Model Loader
# ---------------------------------
def get_embedding_model():

    embedding_model = HuggingFaceEmbeddings(

        model_name=
        "sentence-transformers/all-MiniLM-L6-v2",

        model_kwargs={

            "device": DEVICE,

            "token": HF_TOKEN
        },

        encode_kwargs={

            "batch_size": 32,

            "normalize_embeddings": True
        }
    )

    return embedding_model