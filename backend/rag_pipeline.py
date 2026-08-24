import os
import re
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

class PlantTreatmentRAG:
    def __init__(self, knowledge_base_path=None, persist_directory=None):
        # Get the absolute path to the 'backend' folder (works on any computer)
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        
        # If no paths provided, use the 'backend' folder
        if knowledge_base_path is None:
            knowledge_base_path = os.path.join(backend_dir, "knowledge_base")
        if persist_directory is None:
            persist_directory = os.path.join(backend_dir, "chroma_db")
            
        self.kb_path = knowledge_base_path
        self.persist_dir = persist_directory

        # Initialize embeddings (free, local, no API key)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize Groq LLM
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=os.getenv("GROQ_API_KEY")
        )

        # Load or create vector store
        self.vectorstore = self._load_or_create_vectorstore()
        print(f"✅ RAG pipeline initialized with {self.vectorstore._collection.count()} chunks")

    def _load_pdfs_as_documents(self):
        """Load all PDFs and convert to LangChain Documents with metadata."""
        docs = []
        if not os.path.exists(self.kb_path):
            print(f"⚠️ Knowledge base folder not found: {self.kb_path}")
            return docs

        for file in os.listdir(self.kb_path):
            if file.endswith('.pdf'):
                try:
                    file_path = os.path.join(self.kb_path, file)
                    print(f"📄 Loading PDF: {file}")
                    reader = PdfReader(file_path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    doc = Document(
                        page_content=text,
                        metadata={"source": file}
                    )
                    docs.append(doc)
                    print(f"   ✅ Loaded {len(reader.pages)} pages")
                except Exception as e:
                    print(f"   ❌ Error loading {file}: {e}")
        return docs

    def _load_or_create_vectorstore(self):
        """Load existing Chroma DB or create from PDFs."""
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            print("📂 Loading existing Chroma DB...")
            return Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        else:
            print("🆕 Creating new Chroma DB from PDFs...")
            docs = self._load_pdfs_as_documents()
            if not docs:
                print("⚠️ No PDFs found. Chroma will be empty.")
                return Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_dir
                )

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_documents(docs)
            print(f"📦 Created {len(chunks)} chunks from {len(docs)} documents")

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
            print("💾 Chroma DB saved.")
            return vectorstore

    def _clean_disease_name(self, raw_name):
        """Clean disease name for search queries."""
        name = re.sub(r'^[A-Za-z_]+___', '', raw_name)
        name = name.replace('_', ' ')
        stop_words = {'on','for','the','of','tomato','potato','corn','apple','grape','strawberry','pepper'}
        words = name.split()
        filtered = [w for w in words if w.lower() not in stop_words]
        return ' '.join(filtered) if filtered else name

    def _get_context_with_organic_boost(self, disease_name):
        """Retrieve chunks for the disease, plus extra organic‑focused chunks."""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        main_docs = retriever.invoke(disease_name)
        boosted_query = f"organic {disease_name}"
        boost_docs = retriever.invoke(boosted_query)
        seen = set()
        combined = []
        for doc in main_docs + boost_docs:
            content = doc.page_content[:200]
            if content not in seen:
                seen.add(content)
                combined.append(doc)
        combined = combined[:15]
        context = "\n\n".join(doc.page_content for doc in combined)
        return context

    def query(self, question):
        """Retrieve relevant chunks and use Groq to generate three treatment plans."""
        print(f"🔍 Query: {question}")
        raw = question.replace("What is the treatment for ", "").replace("?", "").strip()
        if " on " in raw:
            disease_raw = raw.split(" on ")[0].strip()
        else:
            disease_raw = raw

        disease_clean = self._clean_disease_name(disease_raw)
        disease_title = disease_clean.title()
        print(f"   Disease: {disease_clean}")

        context = self._get_context_with_organic_boost(disease_clean)
        if not context.strip():
            print("⚠️ No relevant chunks found. Using LLM-only mode.")
            context = "No specific reference information available."

        prompt_template = """
You are a plant pathology expert. Provide three treatment plans for **{disease_name}** on tomatoes.

Reference information (may include both organic and chemical recommendations):
---
{context}
---

**Instructions:**
- For the **ORGANIC** section, **always** include specific organic methods (even if the reference lacks them – use your knowledge). Mention products like neem oil, copper, sulfur, Bacillus subtilis, compost tea, crop rotation, sanitation, etc.
- For the **CHEMICAL** section, list synthetic fungicides (e.g., chlorothalonil, mancozeb) with timing and resistance management.
- For the **PREVENTION** section, cover resistant varieties, spacing, irrigation, and field hygiene.

**Format exactly as:**

###ORGANIC###
1. ...
2. ...

###CHEMICAL###
1. ...
2. ...

###PREVENTION###
1. ...
2. ...
"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()

        try:
            response = chain.invoke({
                "disease_name": disease_title,
                "context": context[:5000]
            })

            organic = ""
            chemical = ""
            prevention = ""

            if "###ORGANIC###" in response and "###CHEMICAL###" in response:
                parts = response.split("###CHEMICAL###")
                organic = parts[0].replace("###ORGANIC###", "").strip()
                if "###PREVENTION###" in parts[1]:
                    chem_prevention = parts[1].split("###PREVENTION###")
                    chemical = chem_prevention[0].strip()
                    prevention = chem_prevention[1].strip() if len(chem_prevention) > 1 else ""
                else:
                    chemical = parts[1].strip()
            else:
                organic_match = re.search(r'###ORGANIC###(.*?)(?=###CHEMICAL###|$)', response, re.DOTALL)
                chemical_match = re.search(r'###CHEMICAL###(.*?)(?=###PREVENTION###|$)', response, re.DOTALL)
                prevention_match = re.search(r'###PREVENTION###(.*?)$', response, re.DOTALL)
                organic = organic_match.group(1).strip() if organic_match else ""
                chemical = chemical_match.group(1).strip() if chemical_match else ""
                prevention = prevention_match.group(1).strip() if prevention_match else ""

            if not organic.strip():
                organic = "**Organic treatment:** Remove infected leaves, apply neem oil or copper-based fungicide, improve air circulation, and rotate crops."
            if not chemical.strip():
                chemical = "**Chemical treatment:** Apply chlorothalonil or mancozeb at first sign, follow label rates, rotate fungicide classes."
            if not prevention.strip():
                prevention = "**Prevention:** Use disease-free seed, practice crop rotation, avoid overhead watering, maintain plant spacing."

            return {
                'organic': organic,
                'chemical': chemical,
                'prevention': prevention
            }

        except Exception as e:
            print(f"⚠️ Groq API error: {e}")
            return self._generate_fallback(disease_title)

    def _generate_fallback(self, disease_name):
        return {
            'organic': f"**Organic / Cultural Treatment for {disease_name}**:\n\n1. Remove infected leaves immediately.\n2. Apply organic fungicide (e.g., copper or sulfur).\n3. Water at the base to reduce moisture.\n4. Ensure proper air circulation by pruning.\n5. Rotate crops every 2-3 years.",
            'chemical': f"**Chemical Treatment for {disease_name}**:\n\n1. Apply appropriate fungicide (e.g., chlorothalonil, mancozeb).\n2. Follow label instructions for application rates.\n3. Rotate chemical classes to prevent resistance.\n4. Apply during cool, dry conditions.",
            'prevention': f"**Prevention for {disease_name}**:\n\n1. Use disease-free seeds or transplants.\n2. Practice crop rotation (2-3 year cycle).\n3. Maintain proper plant spacing for air circulation.\n4. Water at the base to keep foliage dry.\n5. Remove and destroy infected plant debris."
        }

    def get_relevant_chunks(self, question, k=3):
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(self._clean_disease_name(question))
            return [doc.page_content[:500] + "..." for doc in docs]
        except:
            return ["No chunks available."]