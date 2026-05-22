import os

import torch

from dotenv import load_dotenv

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

# ---------------------------------
# Load environment variables
# ---------------------------------
load_dotenv()


# ---------------------------------
# Check CUDA
# ---------------------------------
print("CUDA Available:",
      torch.cuda.is_available())

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
# Hugging Face Token
# ---------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")


# ---------------------------------
# Embedding Model Loader
# ---------------------------------
def get_embedding_model():
    """
    Load embedding model
    with CUDA support.
    """

    embedding_model = HuggingFaceEmbeddings(

        model_name=
        "sentence-transformers/all-MiniLM-L6-v2",

        model_kwargs={

            # GPU usage
            "device": DEVICE,

            # HuggingFace auth
            "token": HF_TOKEN
        },

        encode_kwargs={

            # Better batching
            "batch_size": 32,

            # Normalize embeddings
            "normalize_embeddings": True
        }
    )

    return embedding_model