# LDPTrace

<div align=center>
<img src=./fig/framework.jpg width="50%" ></img>
</div>

This is our Python implementation for the paper:

> Yuntao Du, Yujia Hu, Zhikun Zhang, Ziquan Fang, Lu Chen, Baihua Zheng and Yunjun Gao (2023). LDPTrace: Locally Differentially Private Trajectory Synthesis.  Paper in [arXiv](https://arxiv.org/abs/2302.06180) or [PVLDB](https://www.vldb.org/pvldb/vol16/p1897-gao.pdf). In VLDB'23, Vancouver, Canada, August 28 to September 1, 2023.

See our [blog](https://research.zealscott.com/blog/2023/04/22/LDPTrace/) for the introduction to this work.

## Environment Requirements

- Ubuntu OS
- Python >= 3.8 (Anaconda3 is recommended)
- numpy == 1.21.4

## Dataset

### Dataset Statistics

We conduct our experiments on four benchmark trajectory datasets. The overall statistics are listed below:

| Dataset   | Size      | Average Length | Sampling Interval |
| --------- | --------- | -------------- | ----------------- |
| Oldenburg | 500,000   | 69.75          | 15.6 sec          |
| Porto     | 361,591   | 34.13          | 15 sec            |
| Hangzhou  | 348,144   | 125.02         | 5 sec             |
| Campus    | 1,000,000 | 35.98          | 25 sec            |

### Oldenburg

* Oldenburg is a synthetic dataset simulated by Brinkhoff's network-based moving objects generator. It is based on the map of Oldenburg city, Germany.

* For Oldenburg dataset, please refer to http://iapg.jade-hs.de/personen/brinkhoff/generator/ to generate the synthesized dataset. The setting parameters we used are as follows:
   * obj./time 0 0
   * maximum time: 1000
   * classes: 1 0
   * max. speed div: 50

* After obtaining the raw dataset, it needs to be transformed to the standard input format:

   ```
   #0:
   >0: x_0,y_0; x_1,y_1;...
   #1:
   >0: x_0,y_0; x_1,y_1;...
   #2:
   >0:...
   ...
   ```
   '>0' is a fixed string denoting the start of a trajectory.

   Different format can also work if the type of variable `db` in the code is guaranteed to be `List[Tuple[float, float]]`.
* Locate the dataset into `./LDPTrace/data/` dictionary.

### Porto

* Porto contains taxi traces over 8 months in the city of Porto, Portugal.
* Download the preprocessed dataset ``porto.xz`` from [Google Drive](https://drive.google.com/drive/folders/13bEAx5l2XZhDxurbm482VRF9fYdy_3j7?usp=sharing) or [website_part1](https://zealscott.com/files/datasets/trajectory/porto.7z), and locate them into `./LDPTrace/data/` dictionary.

### Hangzhou

* Hangzhou is a real world trajectory database which consists of the trace of taxis in Hangzhou city, China.
* Due to the **private** nature of Hangzhou dataset, it can not be uploaded to this public repository.

### Campus

* Campus contains 1 million generated trajectories based on the buildings of British Columbia campus.
* Download the preprocessed dataset ``campus.xz`` from [Google Drive](https://drive.google.com/drive/folders/13bEAx5l2XZhDxurbm482VRF9fYdy_3j7?usp=sharing) or [website_part2](https://zealscott.com/files/datasets/trajectory/campus.7z), and locate them into `./LDPTrace/data/` dictionary.

## Reproducibility & Run

Please make sure the data file is in ``./LDPTrace/data/`` dictionary

Here's an example of running LDPTrace:

```python
python main.py --dataset oldenburg --grid_num 6 --max_len 0.9 --epsilon 1.0 --re_syn --multiprocessing
```

LDPTrace will save the synthesized database in ``./LDPTrace/data/DATASET_NAME/`` and output the evaluation metrics.

## Configurations

The running parameters include:

+ --dataset: 
  + 'oldenburg': for Oldenburg dataset
  + 'porto': for Porto dataset
  + 'campus': for Campus dataset
+ --epsilon: privacy budget
+ --grid_num: grid granularity `N`, the spatial map will be decomposed into `N x N` grids. Based on the theoretical analysis in our paper, we recommend `N=6` for Oldenburg, Porto and Campus dataset.
+ --max_len: quantile of estimated max length, the default setting is 0.9
+ --size_factor: reciprocal of query size `r` (i.e., `1/r`), the default setting is 9
+ --query_num: the number of range queries, LDPTrace will output the average query error. The default setting is 200
+ --re_syn: whether to re-synthesize the database. If this parameter is not set, LDPTrace will try to read the saved databased that is synthesized before.
+ --multiprocessing: whether to use multiprocessing in experiments to improve efficiency.

## Acknowledgement

Any scientific publications that use our datasets/codes or mention our work should cite the following paper as the reference:

```
@inproceedings{LDPTrace,
  author    = {Yuntao Du and 
               Yujia Hu and 
               Zhikun Zhang and
               Ziquan Fang and 
               Lu Chen and 
               Baihua Zheng and 
               Yunjun Gao},
  title     = {{LDPTrace}: Locally Differentially Private Trajectory Synthesis},
  booktitle = {{PVLDB}},
  pages     = {1897--1909},
  year      = {2023}
}
```


Nobody guarantees the correctness of the data, its suitability for any particular purpose, or the validity of results based on the use of the data set. The data set may be used for any research purposes under the following conditions:

* The user must acknowledge the use of the data set in publications resulting from the use of the data set.
* The user may not redistribute the data without separate permission.
* The user may not try to deanonymise the data.
* The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from us.
