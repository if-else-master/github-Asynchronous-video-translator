import sys
import os

# 假設 encoder 模組在當前目錄的子目錄中
aa = sys.path.append(os.path.join(os.path.dirname(__file__), 'encoder'))

from encoder.inference import plot_embedding_as_heatmap

print(aa)