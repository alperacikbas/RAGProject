import customtkinter as ctk
from rag import RAGBackend
import threading
import os

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

USER_BUBBLE_COLOR = "#2b2b2b"
MODEL_BUBBLE_COLOR = "#1f1f1f"
USER_TEXT_COLOR = "#E0E0E0"
MODEL_TEXT_COLOR = "#C7C7C7"

class ChatApp(ctk.CTk):
    def __init__(self, backend: RAGBackend):
        super().__init__()

        self.backend = backend

        # --- Ana Pencere Ayarları ---
        self.title("Ders Asistanı (RAG + Gemini)")
        self.geometry("700x600")

        # --- Grid Yapılandırması ---
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # --- Sohbet Alanı (Kaydırılabilir Çerçeve) ---
        self.chat_frame = ctk.CTkScrollableFrame(self)
        self.chat_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # --- Giriş Alanı (Alt Çerçeve) ---
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.entry_box = ctk.CTkEntry(self.input_frame, placeholder_text="Sorunuzu buraya yazın...")
        self.entry_box.grid(row=0, column=0, padx=(0, 10), pady=10, sticky="ew")
        # Enter tuşuna basıldığında da mesajı göndermesi için:
        self.entry_box.bind("<Return>", self.on_send_event)

        self.send_button = ctk.CTkButton(self.input_frame, text="Gönder", command=self.on_send_event)
        self.send_button.grid(row=0, column=1, pady=10, sticky="e")

        # --- Başlangıç (Yükleme) İşlemleri ---
        # Backend'i (RAG zinciri) arayüzü dondurmadan arka planda başlat
        self.start_backend_setup()

    def start_backend_setup(self):
        self.add_message_bubble("Asistan başlatılıyor... Ders notları yükleniyor...", "model")
        # Butonları ve giriş kutusunu kilitliyoruz
        self.set_input_state("disabled")
        
        # setup_rag_chain fonksiyonunu bir thread'de çalıştır
        setup_thread = threading.Thread(target=self.initialize_backend, daemon=True)
        setup_thread.start()

    def initialize_backend(self):
        success = self.backend.setup_rag_chain()
        
        # Kurulum bittiğinde, GUI'yi ana thread üzerinden güncelle
        # .after(0, ...) komutu, fonksiyonun ana GUI thread'inde çalışmasını sağlar
        self.after(0, self.on_backend_ready, success)

    def on_backend_ready(self, success: bool):
        if success:
            self.add_message_bubble("Asistan hazır. Sorularınızı sorabilirsiniz.", "model")
            self.set_input_state("normal")
            self.entry_box.focus()
        else:
            self.add_message_bubble("Hata: Asistan başlatılamadı. 'docs' klasörünü veya API anahtarınızı kontrol edin.", "model")
            # Durum kilitli kalır

    def set_input_state(self, state: str):
        self.entry_box.configure(state=state)
        self.send_button.configure(state=state)

    def on_send_event(self, event=None):
        user_question = self.entry_box.get().strip()
        
        if not user_question:
            return # Boş mesaj göndermeyi engelle
        
        # Kullanıcı mesajını ekle
        self.add_message_bubble(user_question, "user")
        # Giriş kutusunu temizle
        self.entry_box.delete(0, "end")
        
        # Cevap gelene kadar girişleri kilitle
        self.set_input_state("disabled")
        
        # "Düşünüyor..." mesajı ekle
        thinking_label = self.add_message_bubble("Düşünüyor...", "model")
        
        # Cevabı arayüzü dondurmadan almak için yeni bir thread başlat
        response_thread = threading.Thread(target=self.get_model_response, 
                                           args=(user_question, thinking_label), 
                                           daemon=True)
        response_thread.start()

    def get_model_response(self, question: str, thinking_label: ctk.CTkLabel):
        answer = self.backend.ask_question(question)
        
        # Cevap geldiğinde, GUI'yi ana thread üzerinden güncelle
        self.after(0, self.update_answer, answer, thinking_label)

    def update_answer(self, answer: str, thinking_label: ctk.CTkLabel):
        # 'thinking_label' widget'ının metnini güncelle
        thinking_label.configure(text=answer)
        
        # Girişleri tekrar aç
        self.set_input_state("normal")

    def add_message_bubble(self, text: str, role: str) -> ctk.CTkLabel:
        
        if role == "user":
            anchor = "e" # Sağ
            color = USER_BUBBLE_COLOR
            text_color = USER_TEXT_COLOR
        else: # "model"
            anchor = "w" # Sol
            color = MODEL_BUBBLE_COLOR
            text_color = MODEL_TEXT_COLOR
            
        # Mesaj için bir çerçeve oluştur
        frame = ctk.CTkFrame(self.chat_frame, fg_color=color)
        
        # Mesaj etiketini oluştur
        label = ctk.CTkLabel(frame, 
                             text=text, 
                             text_color=text_color, 
                             wraplength=500,
                             justify="left")
        label.pack(padx=10, pady=5)
        
        # Çerçeveyi sohbet alanına yerleştir (sola veya sağa yaslı)
        frame.pack(side="top", anchor=anchor, padx=10, pady=5, fill="x")
        
        # Sohbeti en alta kaydır
        self.after(100, self._scroll_to_bottom)

        return label

    def _scroll_to_bottom(self):
        self.chat_frame._parent_canvas.yview_moveto(1.0)


if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("Hata: Lütfen 'GEMINI_API_KEY' ortam değişkenini ayarlayın.")
    else:
        try:
            backend_instance = RAGBackend()
            app = ChatApp(backend=backend_instance)
            app.mainloop()
        except Exception as e:
            print(f"Uygulama başlatılırken bir hata oluştu: {e}")
            input("Kapatmak için Enter'a basın...")