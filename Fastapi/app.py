from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
# from ollama_integration import generate_legal_response
# from reference import generate_legal_response
# from llama321B import generate_legal_response
# from llama321B_cpu import generate_legal_response
# from llama321B_cpu_deepseek import generate_legal_response
# from llama_merged import generate_legal_response
# from llama3_legal_lora_4epoch import generate_legal_response
from llama321B import generate_legal_response

app = FastAPI()

# Enable CORS for frontend (Node.js or browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query")
async def handle_query(request: Request):
    print("inside endpoint")
    data = await request.json()
    query = data.get("query", "")
    print("callling the functin")
    response = generate_legal_response(query)
    return {"response": response}
