# LCFE-X: Pluggable Local Channel Feature Extraction Model Enhancement Method for Hyperspectral Image Classification
Xiaojun Li, Yi Su, Junping Yao, Hongyang Gu, Yibo Jiao

The code in this toolbox implements the ["LCFE-X: Pluggable Local Channel Feature Extraction Model Enhancement Method for Hyperspectral Image Classification"].

This toolbox includes a proposed module named Local Channel Feature Extraction (LCFE) and a pluggable LCFE-X Model Enhancement Method that can be plug-and-played into Transformer-based hyperspectral image classification basic models. 

For further information, please refer to the paper (https://doi.org/10.1016/j.knosys.2024.112215).


How to use it?
---------------------
We offerd a plagguable LCFE-X model enhancement method for hyperspectral image classification. All you need to do is replace the model in **enhanced_models.py**.

This toolbox provides examples of using **ViT**, **SpectralFormer(SF)** and **MorphFormer(MF)** models on **Pavia University** dataset.

You can directly run **enhanced_models.py** with following parameter settings to reproduce the results. 

Please note that due to differences in the hardware and environment, as well as possible differences in some default parameter settings, the results you obtain may differ slightly from those reported in the paper.

1.	ViT
   
--dataset=PU --tr_percent=0.05 --epoches=300 --patches=7 --group=1 --batch_size=64 --mode=None --backbone=ViT

2.	LCFE-ViT
   
--dataset=PU --tr_percent=0.05 --epoches=300 --patches=7 --group=5 --batch_size=64 --mode=LCFE --backbone=ViT

3.	SF
   
--dataset=PU --tr_percent=0.05 --epoches=300 --patches=7 --group=5 --batch_size=64 --mode=None --backbone=SF

4.	LCFE-SF
   
--dataset=PU --tr_percent=0.05 --epoches=300 --patches=7 --group=5 --batch_size=64 --mode=LCFE --backbone=SF

5.	MF
    
--dataset=PU --tr_percent=0.05 --epoches=300 --patches=7 --group=1 --batch_size=64 --mode=None --backbone=MF

6.	LCFE-MF

--dataset=PU --tr_percent=0.05 --epoches=300 --patches=7 --group=5 --batch_size=64 --mode=LCFE --backbone=MF


Citation
---------------------
Please kindly cite the papers if this code is useful and helpful for your research.

@article{LI2024112215,

title = {LCFE-X: Pluggable Local Channel Feature Extraction model enhancement method for hyperspectral image classification},

journal = {Knowledge-Based Systems},

author = {Xiaojun Li and Yi Su and Junping Yao and Hongyang Gu and Yibo Jiao},

pages = {112215},

year = {2024},

issn = {0950-7051},

doi = {https://doi.org/10.1016/j.knosys.2024.112215}}



@ARTICLE{10141657,

title={A Spectral–Spatial Feature Rotation-Based Ensemble Method for Imbalanced Hyperspectral Image Classification}, 

journal={IEEE Transactions on Geoscience and Remote Sensing}, 

author={Su, Yi and Li, Xiaojun and Yao, Junping and Dong, Chengrong and Wang, Yao},

year={2023},

volume={61},

pages={1-18},

doi={10.1109/TGRS.2023.3282064}}


Requirments
---------------------
PyTorch 1.12 version (CUDA 11.7) in Python 3.8 on Windows system.


Acknowledge
---------------------
Thanks to the following research for their open source code:

[1] Danfeng Hong, Zhu Han, Jing Yao, Lianru Gao, Bing Zhang, Antonio Plaza, and Jocelyn Chanussot. “Spectralformer: Rethinking hyperspectral image classification with transformers,” IEEE Trans. Geosci. Remote Sensing, vol. 60, pp. 1-15, 2022, doi: 10.1109/TGRS.2021.3130716.

[2] S. K. Roy, A. Deria, C. Shah, J. M. Haut, Q. Du, and A. Plaza, “Spectral–Spatial Morphological Attention Transformer for Hyperspectral Image Classification,” IEEE Trans. Geosci. Remote Sensing, vol. 61, pp. 1–15, 2023, doi: 10.1109/TGRS.2023.3242346.


Licensing
---------------------
Copyright (C) 2023 Yi Su.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.


Contact
---------------------
If you have any questions, please feel free to contact us via email (YiSu：email_suyi@163.com) and we will reply as soon as possible.
