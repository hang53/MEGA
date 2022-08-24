# MEGA
The codes of Bootstrapping Informative Graph Augmentation via A Meta Learning Approach

train model:
python3 ./MEGA_train.py --dataset IMDB-MULTI --batch_size 64 --emb_dim 64 --reg_expect 0.4 --pooling_type layerwise

If you find the codes useful, please cite:

@inproceedings{DBLP:conf/ijcai/GaoLQS0Z22,
  author    = {Hang Gao and
               Jiangmeng Li and
               Wenwen Qiang and
               Lingyu Si and
               Fuchun Sun and
               Changwen Zheng},
  editor    = {Luc De Raedt},
  title     = {Bootstrapping Informative Graph Augmentation via {A} Meta Learning
               Approach},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI} 2022, Vienna, Austria, 23-29 July
               2022},
  pages     = {3001--3007},
  publisher = {ijcai.org},
  year      = {2022},
  url       = {https://doi.org/10.24963/ijcai.2022/416},
  doi       = {10.24963/ijcai.2022/416},
  timestamp = {Wed, 27 Jul 2022 16:43:00 +0200},
  biburl    = {https://dblp.org/rec/conf/ijcai/GaoLQS0Z22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
