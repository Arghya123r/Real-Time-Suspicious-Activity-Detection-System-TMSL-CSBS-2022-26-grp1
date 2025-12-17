# Real-Time-Suspicious-Activity-Detection-System-TMSL-CSBS-2022-26-grp1

## Guide to Training

i.Download this dataset: [ucf-crime-dataset](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset)

ii. Run ```pip install requirements.txt``` (virtual environment recommended)

iii.Access Model_Code and training-book-lightweight.ipynb

iv.Copy the **Train** directory folder location and paste it in class cfg **dataset_root**

v. Run All the cells

## Guide to inference
i. copy the **lrcn_ucf_best.pt** into the **Model** Folder

ii.Sample input for the terminal <br>
``` python ActivityDetector.py --input <path to a Video file>```<br>

iii. The output Video will be in the **output** folder

### Information
Realtime video is currently disabled same for webcam view.




