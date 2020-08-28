# Distincitve-attribute Extraction & LSTM-based image captioning

Inference module with pre-trained models
The folder 'images' includes test images from MS COCO datasets*

<!--https://cocodataset.org/-->



### Environment

python3.5
pytorch3.0



### Downloads

Download pre-trained parameters from below and place them in the folder "Pipe_files"

https://drive.google.com/drive/folders/1_wxroYjkgeZkoKADKwzgX1ecDRpUfc-d?usp=sharing



### command

For Korean caption
`python3 DaE_demo.py --is_ko=1 --img_path=[IMAGE_PATH]`

For English caption
`python3 DaE_demo.py --is_ko=0 --img_path=[IMAGE_PATH]`



### Acknowledgement

This work was supported by IITP/MSIT [2017-0-00255, Autonomous digital companion framework and application].