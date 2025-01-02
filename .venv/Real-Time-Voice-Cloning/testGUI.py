import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import torch
import sounddevice as sd
from datetime import datetime
from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from openai import OpenAI
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import pygame

class VoiceCloningGUI:
    def __init__(self, root):     
        self.root = root
        self.root.title("語音克隆與翻譯應用程式 / Voice Cloning & Translation App")
        self.root.geometry("800x800")
                
        
        self.input_path = None
        self.output_dir = None
        self.text_input = None
        self.status_var = tk.StringVar(value="就緒 / Ready")
        self.progress = None
        self.last_generated_wav = None
        self.synthesizer = None
        self.translated_text = tk.StringVar()
        
        # 設定模型路徑
        self.base_path = Path("./.venv/Real-Time-Voice-Cloning")
        self.encoder_path = self.base_path / "encoder/saved_models/encoder.pt"
        self.synthesizer_path = self.base_path / "synthesizer/saved_models/synthesizer.pt"
        self.vocoder_path = self.base_path / "vocoder/saved_models/vocoder.pt"

        # 語言設定
        self.language = tk.StringVar(value="zh")
        self.language_names = {
            "zh": "中文 (Chinese)",
            "en": "英文 (English)",
            "ja": "日文 (Japanese)"
        }
        
        self.create_gui()
        self.initialize_models()
        
    def create_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 標題
        title_text = "語音克隆與翻譯系統 / Voice Cloning & Translation System"
        ttk.Label(main_frame, text=title_text, font=('微軟正黑體', 16, 'bold')).grid(row=0, column=0, columnspan=3, pady=10)
        
        # 輸入檔案選擇
        input_text = "輸入語音檔案 / Input Voice File:"
        ttk.Label(main_frame, text=input_text, font=('微軟正黑體', 10)).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.input_path = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.input_path, width=50).grid(row=1, column=1, pady=5)
        ttk.Button(main_frame, text="瀏覽/Browse", command=self.browse_input).grid(row=1, column=2, padx=5, pady=5)
        
        # 輸出目錄選擇
        output_text = "輸出目錄 / Output Directory:"
        ttk.Label(main_frame, text=output_text, font=('微軟正黑體', 10)).grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_dir = tk.StringVar(value=os.getcwd())
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50).grid(row=2, column=1, pady=5)
        ttk.Button(main_frame, text="瀏覽/Browse", command=self.browse_output).grid(row=2, column=2, padx=5, pady=5)
        
        # 翻譯區域
        translate_frame = ttk.LabelFrame(main_frame, text="語音翻譯 / Voice Translation", padding="5")
        translate_frame.grid(row=3, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Button(translate_frame, text="選擇音訊並翻譯 / Select Audio & Translate", 
                  command=self.translate_audio).grid(row=0, column=0, padx=5, pady=5)
        
        ttk.Label(translate_frame, text="翻譯結果 / Translation:").grid(row=1, column=0, sticky=tk.W, pady=5)
        translation_entry = ttk.Entry(translate_frame, textvariable=self.translated_text, width=50)
        translation_entry.grid(row=1, column=1, columnspan=2, pady=5)
        
        ttk.Button(translate_frame, text="使用翻譯結果 / Use Translation", 
                  command=self.use_translation).grid(row=2, column=0, columnspan=3, pady=5)
        
        # 文字輸入
        text_label = "要合成的文字 / Text to Synthesize:"
        ttk.Label(main_frame, text=text_label, font=('微軟正黑體', 10)).grid(row=4, column=0, sticky=tk.W, pady=5)
        self.text_input = tk.Text(main_frame, height=5, width=50)
        self.text_input.grid(row=4, column=1, columnspan=2, pady=5)
        
        # 語言選擇
        lang_text = "選擇語言 / Select Language:"
        ttk.Label(main_frame, text=lang_text, font=('微軟正黑體', 10)).grid(row=5, column=0, sticky=tk.W, pady=5)
        language_menu = ttk.OptionMenu(
            main_frame, 
            self.language, 
            "zh",
            *self.language_names.keys(),
            command=lambda x: self.update_language_display()
        )
        language_menu.grid(row=5, column=1, columnspan=1, sticky=tk.W, pady=5)
        
        # 狀態顯示
        status_text = "狀態 / Status:"
        ttk.Label(main_frame, text=status_text, font=('微軟正黑體', 10)).grid(row=6, column=0, sticky=tk.W, pady=5)
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=6, column=1, columnspan=2, sticky=tk.W, pady=5)
        
        # 進度條
        self.progress = ttk.Progressbar(main_frame, length=300, mode='determinate')
        self.progress.grid(row=7, column=0, columnspan=3, pady=10)
        
        # 按鈕區域
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=8, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="生成語音/Generate", 
                  command=self.generate_voice).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="播放/Play", 
                  command=self.play_audio).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="停止/Stop", 
                  command=self.stop_audio).pack(side=tk.LEFT, padx=5)

    def initialize_models(self):
        try:
            self.status_var.set("載入模型中... / Loading models...")
            self.root.update()
            
            encoder.load_model(self.encoder_path)
            self.synthesizer = Synthesizer(self.synthesizer_path)
            vocoder.load_model(self.vocoder_path)
            
            self.status_var.set("模型載入成功 / Models loaded")
        except Exception as e:
            messagebox.showerror("Error", f"載入模型失敗 / Model loading failed: {str(e)}")
            self.status_var.set("載入模型錯誤 / Error loading models")

    def translate_audio(self):
        try:
            # 選擇音訊檔案
            file_path = filedialog.askopenfilename(
                filetypes=[
                    ("音訊檔案", "*.mp3 *.wav *.m4a *.mp4"),
                    ("所有檔案", "*.*")
                ]
            )
            
            if not file_path:
                return
                
            self.status_var.set("正在翻譯... / Translating...")
            self.root.update()
            
            # 建立 OpenAI 客戶端
            client = OpenAI(
                api_key = ''
            )
            
            with open(file_path, "rb") as audio_file:
                response = client.audio.translations.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                
                if response:
                    self.translated_text.set(response)
                    print("翻譯結果:", response)
                    self.status_var.set("翻譯完成 / Translation completed")
                else:
                    raise Exception("No translation result received")
                    
        except Exception as e:
            print(f"翻譯錯誤: {str(e)}")
            messagebox.showerror("錯誤", f"翻譯失敗 / Translation failed: {str(e)}")
            self.status_var.set("翻譯失敗 / Translation failed")

    def use_translation(self):
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", self.translated_text.get())
        
    def update_language_display(self):
        selected_lang = self.language.get()
        status_messages = {
            "zh": "就緒 / Ready",
            "en": "Ready",
            "ja": "準備完了"
        }
        self.status_var.set(f"{status_messages[selected_lang]}")
    
    def browse_input(self):
        filetypes = (
            ('音訊檔案 / Audio files', '*.wav *.mp3 *.m4a *.flac'),
            ('所有檔案 / All files', '*.*')
        )
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.input_path.set(filename)
    
    def browse_output(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir.set(directory)
    
    def generate_voice(self):
        try:
            if not self.input_path.get():
                    messagebox.showerror("Error", "請選擇輸入語音檔案 / Please select input file")
                    return
            
            if not self.text_input.get("1.0", tk.END).strip():
                messagebox.showerror("Error", "請輸入要合成的文字 / Please enter text")
                return
            
            self.status_var.set("處理中... / Processing...")
            self.progress['value'] = 20
            self.root.update()
                
            in_fpath = Path(self.input_path.get())
            preprocessed_wav = encoder.preprocess_wav(in_fpath)
                        
            self.status_var.set("建立語音特徵... / Creating voice features...")
            self.progress['value'] = 40
            self.root.update()
            
            embed = encoder.embed_utterance(preprocessed_wav)
            
            self.status_var.set("生成頻譜圖... / Generating spectrogram...")
            self.progress['value'] = 60
            self.root.update()
            
            text = self.text_input.get("1.0", tk.END).strip()
            language = self.language.get()
            
            texts = [f"[{language}]{text}"]
            embeds = [embed]
            specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            
            self.status_var.set("生成波形... / Generating waveform...")
            self.progress['value'] = 80
            self.root.update()
            
            generated_wav = vocoder.infer_waveform(spec)
            generated_wav = np.pad(generated_wav, (0, self.synthesizer.sample_rate), mode="constant")
            generated_wav = encoder.preprocess_wav(generated_wav)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_voice_{timestamp}_{language}.wav"
            output_path = os.path.join(self.output_dir.get(), filename)
            
            sf.write(output_path, generated_wav.astype(np.float32), self.synthesizer.sample_rate)
            
            self.last_generated_wav = (generated_wav, self.synthesizer.sample_rate)
            
            success_msg = f"已生成 / Generated: {filename}"
            self.status_var.set(success_msg)
            self.progress['value'] = 100
            messagebox.showinfo("Success", success_msg)
            
        except Exception as e:
            error_msg = f"錯誤 / Error: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.status_var.set("發生錯誤 / Error occurred")
        finally:
            self.progress['value'] = 0
            

    def play_audio(self):
        if self.last_generated_wav is not None:
            try:
                sd.stop()
                sd.play(*self.last_generated_wav)
                self.status_var.set("播放中... / Playing...")
            except Exception as e:
                messagebox.showerror("Error", f"播放失敗 / Playback failed: {str(e)}")
        else:
            messagebox.showinfo("Info", "尚未生成音訊 / No audio generated")
    
    def stop_audio(self):
        try:
            sd.stop()
            self.status_var.set("已停止 / Stopped")
        except Exception as e:
            messagebox.showerror("Error", f"停止失敗 / Stop failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk() 
    app = VoiceCloningGUI(root)
    root.mainloop()