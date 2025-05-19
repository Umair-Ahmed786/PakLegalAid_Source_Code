from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import re

# Verify GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"Using {torch.cuda.device_count()} GPU(s)")

# --- Legal Classifier Setup ---
# classifier_path = "classifier2"  # Update with your path
classifier_path = "bert-base-uncased-classifier"  # Update with your path
classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
classification_model = AutoModelForSequenceClassification.from_pretrained(classifier_path).to("cuda")

def classify_query(query: str) -> dict:
    """Classifies if query is legal-related with confidence score"""
    inputs = classifier_tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = classification_model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=1)
    legal_prob = probs[0][1].item()
    print("classification prob", legal_prob)
    is_legal = legal_prob > 0.5
    
    return {
        "classification": "LEGAL" if is_legal else "NON-LEGAL",
        "confidence": legal_prob if is_legal else 1 - legal_prob
    }

# --- Vector Store Setup ---
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

# Main QA database
qa_db = FAISS.load_local(
    "Faiss_only_lawyers_db", 
    embeddings=embedding_model, 
    allow_dangerous_deserialization=True
)

# Reference database for citations
reference_db = FAISS.load_local(
    "faiss_legal_db",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# --- LLM Setup ---
model_name = "unsloth/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "base_llama_32_1B",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model = PeftModel.from_pretrained(base_model, "only_lora_1B").to("cuda")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.3,
    top_p=0.9,
    repetition_penalty=1.1,
    device_map="auto",
    torch_dtype=torch.float16
)
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     temperature=0.3,            # Better balance than 0.3
#     top_p=0.85,                # Slightly more focused than 0.9
#     repetition_penalty=1.3,    # Stronger anti-repetition
#     no_repeat_ngram_size=3,    # Prevents 3-word repeats
#     do_sample=True,            # Enables probabilistic sampling
#     device_map="auto",         # Best for multi-GPU
#     torch_dtype=torch.float16, # Uses less VRAM (if GPU supports it)
# )

llm = HuggingFacePipeline(pipeline=pipe)

# --- Prompt Engineering ---
PROMPT_TEMPLATE = """
<|system|>
You are a senior professional Pakistani lawyer with 15+ years providing detailed answers with proper citations. Follow these rules:
1.Carefully understand the client's situation.
2. Answer in Markdown
3. Structure with clear headings
4. Cite sources using:
   - 📘 Book/Act names
   - 🔖 Section numbers
   - 📄 Relevant case law
5. Highlight key points in bold
6. Provide actionable advice when applicable

Context from similar cases that were handled by expert lawyer:
{context}
</|system|>

<|user|>
{question}
</|user|>

<|assistant|>
""" 

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
rag_chain = LLMChain(llm=llm, prompt=prompt, verbose=False)

def extract_references(query, k=3):
    """Extracts formatted references from the reference database"""
    similar_docs = reference_db.similarity_search(query, k=k)
    citations = []
    
    for doc in similar_docs:
        meta = {
            "book": re.search(r"Book:\s*(.+)", doc.page_content),
            "chapter_number": re.search(r"Chapter Number:\s*(.+)", doc.page_content),
            "chapter_title": re.search(r"Chapter Title:\s*(.+)", doc.page_content),
            "section": re.search(r"Section:\s*(.+)", doc.page_content),
            "heading": re.search(r"Heading:\s*(.+)", doc.page_content)
        }
        
        citation_parts = []
        if meta["book"] and meta["book"].group(1):
            citation_parts.append(f"📘 Book: {meta['book'].group(1).strip()}")
        if meta["chapter_title"] and meta["chapter_title"].group(1):
            citation_parts.append(f"📄 Chapter: {meta['chapter_title'].group(1).strip()}")
        if meta["section"] and meta["section"].group(1):
            citation_parts.append(f"🔖 Section: {meta['section'].group(1).strip()}")
        
        if citation_parts:
            citations.append(" • " + " | ".join(citation_parts))
    
    return "\n\n".join(citations) if citations else None


import re

def extract_assistant_response(full_text: str) -> str:
    """Extracts only the assistant's response from the full output"""
    # Try to find the <|assistant|> tag
    if "<|assistant|>" in full_text:
        return full_text.split("<|assistant|>")[-1].strip()
    
    # Fallback: Try to find the last generated part
    parts = re.split(r"<\|[a-z]+\|>", full_text)
    if len(parts) > 1:
        return parts[-1].strip()
    
    return full_text.strip()


def generate_legal_response(query: str, k: int = 4) -> dict:
    try:
        # Classification
        print('inside generate function')
        classification = classify_query(query)
        if classification["classification"] == "NON-LEGAL":
            return {
                "answer": "Non-Legal",
                "sources": None,
                # "classification": classification
            }

        # Retrieval
        similar_docs = qa_db.similarity_search(query, k=k)
        context = "\n".join([
            f"**Similar Case {i+1}:**\n{doc.page_content.split('✅ Answer:')[-1].strip()}"
            for i, doc in enumerate(similar_docs)
        ])
        print("contxt: ",context)

        # Generation
        full_response = rag_chain.run({"context": context, "question": query})
        print("respnse of model: ",full_response)
        
        # Extract just the assistant's answer
        assistant_answer = full_response.split("<|assistant|>")[-1].strip() if "<|assistant|>" in full_response else full_response
        
        references = extract_references(query, k=3)
        full_response = assistant_answer + "\n\nReferences:\n\n" + references
        print("full response: ",full_response)

        return {
            "answer": full_response,  # Only the assistant's response
            "sources": [doc.metadata for doc in similar_docs],
            "classification": classification,
            # "references": references
        }

    except Exception as e:
        return {
            "error": str(e),
            "answer": "An error occurred while processing your query",
            "sources": None
        }
    finally:
        torch.cuda.empty_cache()
# FastAPI Integration (keep your existing app setup)