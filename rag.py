import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

class RAGBackend:
    def __init__(self, pdf_directory="docs", api_key=None):

        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY") 
        
            if not api_key:
                raise ValueError("GEMINI_API_KEY ortam değişkeni ayarlanmamış veya anahtar sağlanmamış.")
        
            
        self.api_key = api_key
        self.pdf_directory = pdf_directory
        self.rag_chain = None
        print("RAGBackend başlatıldı.")

    def setup_rag_chain(self):
        try:
            print("PDF'ler yükleniyor...")
            loader = PyPDFDirectoryLoader(self.pdf_directory)
            docs = loader.load()
            if not docs:
                print(f"Hata: '{self.pdf_directory}' klasöründe PDF bulunamadı.")
                return False

            print("Dokümanlar parçalanıyor...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            print("Yerel Embedding modeli yükleniyor (Bu işlem ilk seferde uzun sürebilir)...")
            
            embeddings = HuggingFaceEmbeddings(
                model_name="paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}
            )

            print("Vektör veritabanı YEREL olarak oluşturuluyor (Bu işlem PDF sayısına göre zaman alabilir)...")
            vectorstore = FAISS.from_documents(splits, embeddings)
            retriever = vectorstore.as_retriever()
            print("Vektör veritabanı başarıyla kuruldu.")

            print("LLM (Gemini Sohbet Modeli) ayarlanıyor...")

            model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3,
                google_api_key=self.api_key 
            )
            
            prompt_template = """
            Sen bir üniversite ders asistanısın. 
            Sana verilen 'Bağlam'ı kullanarak 'Soru'yu detaylı ve doğru bir şekilde cevapla.
            Cevap 'Bağlam'da yer almıyorsa, "Bu bilgi elimdeki ders notlarında bulunmuyor." de.
            Asla bağlam dışı bilgi verme veya bir şey uydurma.

            Bağlam: {context}
            Soru: {input}
            Cevap:
            """
            prompt = ChatPromptTemplate.from_template(prompt_template)

            question_answer_chain = create_stuff_documents_chain(model, prompt)
            self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            print("RAG zinciri başarıyla kuruldu. Asistan hazır.")
            return True
        
        except Exception as e:
            print(f"RAG zinciri kurulurken kritik hata: {e}")
            return False

    def ask_question(self, question: str) -> str:
        if not self.rag_chain:
            return "Hata: RAG zinciri henüz hazır değil."
        
        try:
            print(f"Soru alindi: {question}")
            response = self.rag_chain.invoke({"input": question})
            print("Cevap üretildi.")
            return response['answer']
        except Exception as e:
            print(f"Model sorgulanırken hata: {e}")
            return "Cevap üretirken bir hatayla karşılaştım."