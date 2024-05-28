# ESCM
## Code and data for ACL 2024: An Iterative Associative Memory Model for Empathetic Response Generation.

## Environment
```code
CUDA Version==11.7
python==3.10.13
torch==2.0.1
```
## Environment Installation
To run this code, you should：
1. Clone the project from github.
```sh
git clone https://github.com/zhouzhouyang520/IAMM.git
```
2. Enter the project root directory
```sh
cd IAMM/
```
3. Download the data, and put it into the project root directory. [Baidu cloud link](https://pan.baidu.com/s/18egWGxCLRbC6H5G7XCIDVg?pwd=8suu) with Code: 8suu or [Google cloud link](https://drive.google.com/drive/folders/19g8zZ_TtxZJGRP8hazAEP_MgKy9d-Ipg?usp=sharing)
```sh
unzip ED.zip
mkdir data
mv ED data
mkdir vectors
mv glove.6B.300d.txt vectors
```

4. Install required packages
```sh
pip install -r requirements.txt
```
The project mainly referenced the code from [CEM](https://github.com/Sahandfer/CEM) and [ESCM](https://github.com/zhouzhouyang520/EmpatheticDialogueGeneration_ESCM). If there are issues during installation, you can try using the environments from these two projects.

## Model Training
```sh
out_file=iamm.log 
nohup python main.py --model iamm --cuda --device_id 7 --pointer_gen --word_topk 5 --ctx_topk 15 --cs_topk 5 > $out_file 2>&1 &
```
## Model Test
```sh
out_file=iamm_test.log 
nohup python main.py --model iamm --test --test_model_name IAMM_19_9496.3010_0.0268 --cuda --device_id 7 --pointer_gen --batch_size 48 --word_topk 5 --ctx_topk 15 --cs_topk 5 --hgnn_hidden 300 --hgnn_out 50 > $out_file 2>&1 &
```

If this work is helpful, please kindly cite as:
```code
@inproceedings{
anonymous2024an,
title={An Iterative Associative Memory Model for Empathetic Response Generation},
author={Zhou Yang and Zhaochun Ren and Yufeng Wang and Haizhou Sun and Chao Chen and Xiaofei Zhu and Xiangwen Liao},
booktitle={The 62nd Annual Meeting of the Association for Computational Linguistics},
year={2024},
url={https://openreview.net/forum?id=VemvrNidxs}
}
```
