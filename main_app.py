import customtkinter as ctk
from rag import RAGBackend
import threading
import os

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# --- Renk Tanımlamaları ---
USER_BUBBLE_COLOR = "#2b5278"  # Maviye yakın bir ton
MODEL_BUBBLE_COLOR = "#363636" # Koyu Gri
USER_TEXT_COLOR = "#FFFFFF"
MODEL_TEXT_COLOR = "#E0E0E0"

# Sistem Mesajları (Ortalanmış)
SYSTEM_BUBBLE_COLOR = "#242424" # Çok koyu gri
SYSTEM_TEXT_COLOR = "#909090" # Soluk gri

class ChatApp(ctk.CTk):
    def __init__(self, backend: RAGBackend):
        super().__init__()

        self.backend = backend

        # --- Ana Pencere Ayarları ---
        self.title("Ders Asistanı")
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
        self.entry_box.bind("<Return>", self.on_send_event)

        self.send_button = ctk.CTkButton(self.input_frame, text="Gönder", command=self.on_send_event)
        self.send_button.grid(row=0, column=1, pady=10, sticky="e")

        # --- Başlangıç (Yükleme) İşlemleri ---
        self.start_backend_setup()

    def start_backend_setup(self):
        # --- ESTETİK DEĞİŞİKLİK ---
        # 'model' yerine 'system' rolü ile çağırıyoruz.
        self.add_message_bubble("Asistan başlatılıyor... Ders notları yükleniyor...", "system")
        
        self.set_input_state("disabled")
        
        setup_thread = threading.Thread(target=self.initialize_backend, daemon=True)
        setup_thread.start()

    def initialize_backend(self):
        success = self.backend.setup_rag_chain()
        self.after(0, self.on_backend_ready, success)

    def on_backend_ready(self, success: bool):
        if success:
            # --- ESTETİK DEĞİŞİKLİK ---
            # 'model' yerine 'system' rolü ile çağırıyoruz.
            self.add_message_bubble("Asistan hazır. Sorularınızı sorabilirsiniz.", "system")
            self.set_input_state("normal")
            self.entry_box.focus()
        else:
            # --- ESTETİK DEĞİŞİKLİK ---
            # 'model' yerine 'system' rolü ile çağırıyoruz.
            self.add_message_bubble("Hata: Asistan başlatılamadı. 'docs' klasörünü veya API anahtarınızı kontrol edin.", "system")

    def set_input_state(self, state: str):
        self.entry_box.configure(state=state)
        self.send_button.configure(state=state)

    def on_send_event(self, event=None):
        user_question = self.entry_box.get().strip()
        
        if not user_question:
            return 
        
        # Kullanıcı mesajı ('user' rolü)
        self.add_message_bubble(user_question, "user")
        
        self.entry_box.delete(0, "end")
        self.set_input_state("disabled")
        
        # "Düşünüyor..." mesajı ('model' rolü)
        # Bu bir sistem mesajı değil, modelin cevabı olduğu için 'model' olarak kalır.
        thinking_label = self.add_message_bubble("Düşünüyor...", "model")
        
        response_thread = threading.Thread(target=self.get_model_response, 
                                           args=(user_question, thinking_label), 
                                           daemon=True)
        response_thread.start()

    def get_model_response(self, question: str, thinking_label: ctk.CTkLabel):
        try:
            answer = self.backend.ask_question(question)
        except Exception as e:
            print("get_model_response içinde hata:", e)
            answer = "Bir hata oluştu."

        self.after(0, self.update_answer, answer, thinking_label)


    def update_answer(self, answer: str, thinking_label: ctk.CTkLabel):
        thinking_label.configure(text=answer)
        self.set_input_state("normal")

    # --- ESTETİK DEĞİŞİKLİKLERİN UYGULANDIĞI YER ---
    def add_message_bubble(self, text: str, role: str) -> ctk.CTkLabel:
        
        if role == "user":
            anchor = "e" # Sağ
            color = USER_BUBBLE_COLOR
            text_color = USER_TEXT_COLOR
            justify = "right"
            fill_type = "none" # Genişliği doldurma
            padx = (50, 10) # Sol duvardan 50, sağdan 10 boşluk
        elif role == "model":
            anchor = "w" # Sol
            color = MODEL_BUBBLE_COLOR
            text_color = MODEL_TEXT_COLOR
            justify = "left"
            fill_type = "none" # Genişliği doldurma
            padx = (10, 50) # Sol duvardan 10, sağdan 50 boşluk
        else: # role == "system"
            anchor = "center" # Orta
            color = SYSTEM_BUBBLE_COLOR
            text_color = SYSTEM_TEXT_COLOR
            justify = "center"
            fill_type = "x" # Tüm genişliği kapla
            padx = (10, 10) # Kenarlardan eşit boşluk
            
        # Mesaj için bir çerçeve oluştur
        # Köşeleri yuvarlatıyoruz
        frame = ctk.CTkFrame(self.chat_frame, fg_color=color, corner_radius=10)
        
        # Mesaj etiketini oluştur
        label = ctk.CTkLabel(frame, 
                             text=text, 
                             text_color=text_color, 
                             wraplength=500, # Metin kaydırma
                             justify=justify) # Metin hizalama
        label.pack(padx=10, pady=5)
        
        # Çerçeveyi sohbet alanına yerleştir (değişiklikler burada)
        # fill=fill_type ve padx=padx eklendi
        frame.pack(side="top", anchor=anchor, padx=padx, pady=5, fill=fill_type)
        
        self.after(100, self._scroll_to_bottom)

        return label

    def _scroll_to_bottom(self):
        self.after(10, self.chat_frame._parent_canvas.yview_moveto, 1.0)


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