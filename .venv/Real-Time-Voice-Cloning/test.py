import torchaudio
import numpy as np
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
import soundfile as sf
from pathlib import Path
import torch
import logging
from datetime import datetime

def setup_logging():
    """設置日誌記錄"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        filename=f'voice_clone_{timestamp}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def clone_voice(wav_file_path, text, target_sr=40000):
    """
    語音克隆主函數
    
    Parameters:
        wav_file_path (str): 輸入音頻文件路徑
        text (str): 要合成的文本
        target_sr (int): 目標採樣率，默認48000Hz
    """
    setup_logging()
    
    try:
        # 1. 載入並預處理音頻
        logging.info(f"正在載入音頻文件: {wav_file_path}")
        wav, sample_rate = torchaudio.load(wav_file_path)
        logging.info(f"原始音頻形狀: {wav.shape}, 採樣率: {sample_rate}")
        
        # 重採樣處理
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            wav = resampler(wav)
            logging.info(f"重採樣後音頻形狀: {wav.shape}")
        
        # 確保音頻是單聲道
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            logging.info("已轉換為單聲道")
        
        # 限制音頻長度（5秒）
        max_length = 5 * target_sr
        if wav.shape[1] > max_length:
            wav = wav[:, :max_length]
            logging.info(f"已截斷音頻至 {max_length} 採樣點")
        
        wav = wav.squeeze().numpy().astype(np.float32)
        logging.info(f"最終預處理後音頻形狀: {wav.shape}")

        # 2. 載入模型
        logging.info("正在載入模型...")
        base_path = Path("D:\專案\github Asynchronous-video-translator\Asynchronous-video-translator\.venv\Real-Time-Voice-Cloning")
        encoder_path = base_path / "encoder/saved_models/encoder.pt"
        synthesizer_path = base_path / "synthesizer/saved_models/synthesizer.pt"
        vocoder_path = base_path / "vocoder/saved_models/vocoder.pt"

        # 檢查模型文件
        for path in [encoder_path, synthesizer_path, vocoder_path]:
            if not path.exists():
                raise FileNotFoundError(f"找不到模型文件: {path}")

        encoder.load_model(encoder_path)
        synthesizer = Synthesizer(synthesizer_path)
        vocoder.load_model(vocoder_path)
        
        # 3. 預處理音頻
        logging.info("正在預處理音頻...")
        preprocessed_wav = encoder.preprocess_wav(wav, target_sr)
        logging.info(f"預處理後音頻形狀: {preprocessed_wav.shape}")
        
        # 4. 計算嵌入向量
        logging.info("正在計算嵌入向量...")
        embed = encoder.embed_utterance(preprocessed_wav)
        logging.info(f"嵌入向量形狀: {embed.shape}")
        
        # 5. 生成語音
        logging.info("正在合成語音...")
        specs = synthesizer.synthesize_spectrograms([text], [embed])
        logging.info(f"生成的頻譜圖形狀: {specs[0].shape}")
        
        # 6. 生成波形
        logging.info("正在生成波形...")
        generated_wav = vocoder.infer_waveform(specs[0])
        generated_wav = generated_wav.astype(np.float32)
        logging.info(f"生成的波形形狀: {generated_wav.shape}")
        
        # 7. 保存輸出
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_path / "voice"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"output_{timestamp}.wav"
        
        sf.write(output_path, generated_wav, target_sr)
        logging.info(f"已成功保存生成的語音到 {output_path}")
        
        return output_path
        
    except Exception as e:
        logging.error(f"語音克隆過程中發生錯誤: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        wav_file_path = "D:/專案/Real-Time-Voice-Cloning/.venv/Real-Time-Voice-Cloning/Bo2.wav"
        text = "ohohohohohohohohohohohohohohohohohohohohohohohohohohohohohohohohohohohohohohohohohohoh"
        
        output_path = clone_voice(wav_file_path, text)
        print(f"成功生成語音：{output_path}")
    except Exception as e:
        print(f"語音克隆失敗：{str(e)}")

# import argparse
# import os
# from pathlib import Path
# import librosa
# import numpy as np
# import soundfile as sf
# import torch
# from encoder import inference as encoder
# from encoder.params_model import model_embedding_size as speaker_embedding_size
# from synthesizer.inference import Synthesizer
# from utils.argutils import print_args
# from utils.default_models import ensure_default_models
# from vocoder import inference as vocoder
# import logging
# from datetime import datetime

# def setup_logging(output_dir):
#     """設置日誌記錄"""
#     log_dir = Path(output_dir) / "logs"
#     log_dir.mkdir(parents=True, exist_ok=True)
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     logging.basicConfig(
#         filename=log_dir / f'voice_clone_{timestamp}.log',
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )

# def clone_voice(in_fpath, text, output_dir, args):
#     """
#     執行語音克隆
    
#     Args:
#         in_fpath (Path): 輸入音頻文件路徑
#         text (str): 要合成的文本
#         output_dir (Path): 輸出目錄路徑
#         args: 命令行參數
#     """
#     try:
#         # 確保輸出目錄存在
#         output_dir = Path(output_dir)
#         output_dir.mkdir(parents=True, exist_ok=True)
        
#         # 設置日誌
#         setup_logging(output_dir)
#         logging.info(f"開始處理音頻文件: {in_fpath}")
        
#         # 載入模型
#         print("正在載入模型...")
#         ensure_default_models(Path("saved_models"))
#         encoder.load_model(args.enc_model_fpath)
#         synthesizer = Synthesizer(args.syn_model_fpath)
#         vocoder.load_model(args.voc_model_fpath)
        
# #         base_path = Path("D:/專案/Real-Time-Voice-Cloning/.venv/Real-Time-Voice-Cloning")
# #         encoder_path = base_path / "encoder/saved_models/encoder.pt"
# #         synthesizer_path = base_path / "synthesizer/saved_models/synthesizer.pt"
# #         vocoder_path = base_path / "vocoder/saved_models/vocoder.pt"

#         # 預處理音頻
#         print("正在處理輸入音頻...")
#         preprocessed_wav = encoder.preprocess_wav(in_fpath)
        
#         # 生成嵌入向量
#         print("正在生成聲音特徵...")
#         embed = encoder.embed_utterance(preprocessed_wav)
        
#         # 生成頻譜圖
#         print("正在生成頻譜圖...")
#         specs = synthesizer.synthesize_spectrograms([text], [embed])
#         spec = specs[0]
        
#         # 生成波形
#         print("正在合成語音...")
#         generated_wav = vocoder.infer_waveform(spec)
        
#         # 後處理
#         generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
#         generated_wav = encoder.preprocess_wav(generated_wav)
        
#         # 生成輸出文件名
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_fname = output_dir / f"generated_voice_{timestamp}.wav"
        
#         # 保存音頻
#         sf.write(str(output_fname), generated_wav.astype(np.float32), synthesizer.sample_rate)
#         print(f"\n已保存生成的語音到: {output_fname}")
        
#         # 如果需要播放音頻
#         if not args.no_sound:
#             try:
#                 import sounddevice as sd
#                 sd.stop()
#                 sd.play(generated_wav, synthesizer.sample_rate)
#                 print("正在播放生成的語音...")
#             except Exception as e:
#                 print(f"無法播放音頻: {str(e)}")
        
#         return output_fname
        
#     except Exception as e:
#         logging.error(f"語音克隆過程中發生錯誤: {str(e)}", exc_info=True)
#         raise

# def main():
#     parser = argparse.ArgumentParser(description="語音克隆工具")
#     base_path = Path("D:/專案/Real-Time-Voice-Cloning/.venv/Real-Time-Voice-Cloning")
    
#     # 原有的參數
#     parser.add_argument("-e", "--enc_model_fpath", type=Path,
#                       default=base_path / "encoder/saved_models/encoder.pt",
#                       help="編碼器模型路徑")
#     parser.add_argument("-s", "--syn_model_fpath", type=Path,
#                       default=base_path / "synthesizer/saved_models/synthesizer.pt",
#                       help="合成器模型路徑")
#     parser.add_argument("-v", "--voc_model_fpath", type=Path,
#                       default=base_path / "vocoder/saved_models/vocoder.pt",
#                       help="聲碼器模型路徑")
#     parser.add_argument("--cpu", action="store_true",
#                       help="強制使用CPU進行處理")
#     parser.add_argument("--no_sound", action="store_true",
#                       help="不播放音頻")
#     parser.add_argument("--seed", type=int, default=None,
#                       help="隨機數種子")
    
#     # 預設輸入值
#     parser.add_argument("--input", type=Path, default=Path("D:/專案/Real-Time-Voice-Cloning/.venv/Real-Time-Voice-Cloning/Bo2.wav"),
#                       help="輸入的WAV文件路徑")
#     parser.add_argument("--output_dir", type=Path, default=Path("D:/專案/Real-Time-Voice-Cloning/.venv/Real-Time-Voice-Cloning/voice"),
#                       help="輸出目錄路徑")
#     parser.add_argument("--text", type=str, default="我是天才我是天才我是天才我是天才我是天才我是天才我是天才我是天才我是天才我是天才",
#                       help="要合成的文字")
    
#     args = parser.parse_args()
    
#     # 處理GPU設置
#     if args.cpu:
#         os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
#     if torch.cuda.is_available():
#         device_id = torch.cuda.current_device()
#         gpu_properties = torch.cuda.get_device_properties(device_id)
#         print(f"使用GPU {device_id} ({gpu_properties.name}) 進行處理\n")
#     else:
#         print("使用CPU進行處理\n")
    
#     try:
#         output_path = clone_voice(args.input, args.text, args.output_dir, args)
#         print(f"\n處理完成！輸出文件：{output_path}")
#     except Exception as e:
#         print(f"錯誤：{str(e)}")
#         return 1
    
#     return 0

# if __name__ == "__main__":
#     exit_code = main()
#     exit(exit_code)


# #改造成一個GUI畫面，包含加入原本WAV檔位置、輸出的文字、輸出到電腦的哪個資料夾