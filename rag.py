# rag.py (güncellenmiş, sorunsuz sürüm)

import os
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class RAGBackend:
    def __init__(self, pdf_directory="docs", api_key=None):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("=" * 50)
            print("!!! HATA: GEMINI_API_KEY ortam değişkeni bulunamadı!")
            print("Lütfen terminalde: set GEMINI_API_KEY=... şeklinde ayarlayın.")
            print("=" * 50)
            raise ValueError("GEMINI_API_KEY ortam değişkeni ayarlanmamış.")
        self.api_key = api_key
        self.pdf_directory = pdf_directory

        # Zincirler
        self.query_transform_chain = None
        self.base_retriever = None
        self.question_answer_chain = None

        print("RAGBackend başlatıldı.")

    def setup_rag_chain(self):
        try:
            # --- 1. AŞAMA: PDF yükleme ---
            print("PDF'ler yükleniyor...")
            loader = PyPDFDirectoryLoader(self.pdf_directory)
            docs = loader.load()
            if not docs:
                print(f"Hata: '{self.pdf_directory}' klasöründe PDF bulunamadı.")
                return False

            print("Dokümanlar parçalanıyor...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)

            # --- 2. AŞAMA: Embedding + FAISS ---
            print("Yerel Embedding modeli yükleniyor...")
            embeddings = HuggingFaceEmbeddings(
                model_name="paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={"device": "cpu"},
            )

            print("Vektör veritabanı YEREL olarak oluşturuluyor...")
            faiss_index = FAISS.from_documents(splits, embeddings)
            self.base_retriever = faiss_index.as_retriever()
            print("Vektör veritabanı başarıyla kuruldu.")

            # --- 3. AŞAMA: LLM ---
            print("LLM (Gemini) ayarlanıyor...")
            model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3,
                google_api_key=self.api_key,
            )

            # --- 4. Sorgu dönüştürme chain ---
            transform_prompt_template = """
            Bir kullanıcı sohbet sorusu sordu. Bu soruyu bir vektör veritabanında aramak için 
            en uygun, optimize edilmiş 'anahtar kelime' sorgusuna dönüştür.
            Sadece optimize edilmiş sorguyu döndür.

            Orijinal Soru: {input}
            Optimize Edilmiş Sorgu:"""

            transform_prompt = ChatPromptTemplate.from_template(
                transform_prompt_template
            )

            self.query_transform_chain = (
                transform_prompt | model | StrOutputParser()
            )

            # --- 5. Soru + bağlam + LLM cevabı chain ---
            gen_prompt_template = """
            Sen bir üniversite ders asistanısın. 
            Aşağıdaki 'Bağlam' içindeki bilgilere dayanarak 'Soru'yu cevapla.
            Bağlamda olmayan bilgiyi ASLA uydurma.
            Bağlamda yoksa: "Bu bilgi elimdeki ders notlarında bulunmuyor." de.

            Bağlam:
            {context}

            Soru:
            {input}

            Cevap:
            """

            gen_prompt = ChatPromptTemplate.from_template(gen_prompt_template)

            # ❗ create_stuff_documents_chain KALDIRILDI
            self.question_answer_chain = (
                gen_prompt | model | StrOutputParser()
            )

            print("AKILLI RAG zinciri başarıyla kuruldu. Asistan hazır.")
            return True

        except Exception as e:
            print(f"RAG zinciri kurulurken kritik hata: {e}")
            return False

    # ----------------- Yardımcı -----------------

    def _normalize_to_str(self, obj: Any) -> str:
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            for k in ("output_text", "answer", "result", "text"):
                if k in obj:
                    return str(obj[k])
            return str(obj)
        return str(obj)

    def _safe_to_text(self, doc):
        try:
            if hasattr(doc, "page_content"):
                return str(doc.page_content)
            if isinstance(doc, dict):
                return str(doc.get("page_content", str(doc)))
            if isinstance(doc, str):
                return doc
            return str(doc)
        except:
            return ""

    # ----------------- SORU SORMA -----------------

    def ask_question(self, question: str) -> str:
        print("\n--- ASK QUESTION ÇAĞRILDI ---")
        print("Kullanıcı sorusu:", question)

        try:
            # 1) Sorgu optimize et
            print("[1] Sorgu optimize ediliyor...")
            transformed_raw = self.query_transform_chain.invoke(
                {"input": question}
            )
            transformed_query = (
                self._normalize_to_str(transformed_raw).strip()
            )
            if not transformed_query:
                transformed_query = question

            print("  Transform:", transformed_query)

            # 2) Retriever çağır
            print("[2] Retriever çalışıyor (invoke)...")
            docs_raw = self.base_retriever.invoke(transformed_query)
            print("  Raw retriever çıktısı:", docs_raw)

            docs = (
                docs_raw
                if isinstance(docs_raw, list)
                else docs_raw.get("documents", [])
                if isinstance(docs_raw, dict)
                else []
            )

            print(f"  {len(docs)} doküman bulundu.")

            # 3) Bağlam oluştur
            print("[3] Bağlam oluşturuluyor...")
            parts = [self._safe_to_text(d) for d in docs]
            context_text = "\n\n".join(parts)
            print("[3] Bağlam oluşturuldu.")

            # 4) LLM çağırılıyor
            print("[4] LLM çağırılıyor...")
            result = self.question_answer_chain.invoke(
                {
                    "input": question,
                    "context": context_text,
                }
            )

            print("[4] LLM Raw:", result)
            final_answer = self._normalize_to_str(result)

            print("[5] Final:", final_answer)
            return final_answer

        except Exception as e:
            print("!!! ask_question() içinde HATA !!!")
            print(e)
            return "Bir hata oluştu."
