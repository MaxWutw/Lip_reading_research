# Lip_reading_research
## 摘要
本研究為利用類神經網路進行唇語辨識，我們將唇語辨識視為 Seq2Seq 任務，輸入為圖片的
序列，輸出為字元 (character) 的序列。我們分為四個實驗，其中會使用同一種資料集但是
切割資料集方式不同分別稱為 Overlapped 和 Unseen，第一個實驗我們嘗試使用 3D CNN 加
上 GRU，第二個實驗並在每個影格的圖片進行提取並使用 2 維卷積加上 GRU 來進行訓
練，第三個實驗是使用 3D CNN + ResNet + Bi-GRU，第四個實驗為 3D CNN + ResNet + 
Transformer。我們的字元錯誤率 (Character Error Rate, CER) 在 Overlapped 中最低可以得到
2.4%，在 Unseen 中最低可以得到 8.4%。
## 環境要求
在Python3 以上皆可

## Each file description
main.py : For training and validation<br>
LCANet.py : modify from LCANet model<br>
LipNet.py : modify from LipNet model<br>
MH_model.py : 3D CNN + ResNet + Multi-Head Attention<br>
cvtransforms.py : Data augmentation<br>
dataset.py : Custom Dataset<br>
Data_preprocessing : Video to time series image, data preprocessing<br>
options.py : Hyperparameters<br>
