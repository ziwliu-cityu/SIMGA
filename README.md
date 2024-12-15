# SIMGA: Selective Gated Mamba for Recommendation System
This is the implementation of the submission "SIGMA: Selective Gated Mamba for Sequential Recommendation".
## Configuration of the environment
The hardware and software we used are listed below to facilitate the environment's configuration. The detailed environment setting can be found in the `requirements.txt`. You can use pip install to reproduce the environment.
- Hardware:
  - GPU: one NVIDIA L4
  - CUDA: 11.8
- Software:
  - Python: 3.10.13
  - Pytorch: 2.1.1 + cu118
- Usage
  - Install Causal Conv1d
    - `pip install causal-conv1d==1.1.3.post1`
  - Install Recbole
    - `pip install recbole==1.2.0`
  - Install Mamba
    - `pip install mamba-ssm==1.1.4`
A detailed configuration process in Colab can be found in the `RecMamba.ipynb`
##  Datasets
The procedures for preprocessing the datasets are listed as follows:
- The raw datasets should be preprocessed using the Conversion tool provided by `https://github.com/RUCAIBox/RecSysDatasets`. After you acquire the atomic files, please put them into `dataset/<Amazon_Fashion/Amazon_Sports_and_Outdoors/Amazon_Video_Games/amazon-beauty/ml-1m/yelp>`. The Yelp Dataset can be found at `https://www.yelp.com/dataset`; The Amazon datasets can be found at `https://cseweb.ucsd.edu/jmcauley/datasets.html\#amazon_reviews`; The MovieLens-1M dataset can be found at `https://grouplens.org/datasets/movielens/`.
- Or you can directly download the atomic files of these datasets using the Baidu disk link provided by Recbole: `https://github.com/RUCAIBox/RecSysDatasets`.

For the procedure of filtering the cold-start users and items, please find the corresponding part in the `model/config.yaml`

## Model Training
- You can directly run the `model/run.py` to reproduce the training procedure.
- For evaluation on grouped users, please refer to `model/run_custrainer.py`.

## Citation
If you found the code and the paper are useful, please kindly cite our paper:
```bibtex
@article{liu2024bidirectional,
  title={Bidirectional gated mamba for sequential recommendation},
  author={Liu, Ziwei and Liu, Qidong and Wang, Yejing and Wang, Wanyu and Jia, Pengyue and Wang, Maolin and Liu, Zitao and Chang, Yi and Zhao, Xiangyu},
  journal={arXiv preprint arXiv:2408.11451},
  year={2024}
}
```
