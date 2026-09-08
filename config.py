from dotenv import load_dotenv
import os

load_dotenv()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

QWEN3_AGENT_URL = os.getenv("QWEN3_AGENT_URL")
BGE_EMBED_URL = os.getenv("BGE_EMBED_URL")
QWEN_VL_URL = os.getenv("QWEN_VL_URL")
QWEN_CODER_URL = os.getenv("QWEN_CODER_URL")