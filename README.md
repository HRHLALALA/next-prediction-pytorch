# next-prediction.pytorch
 
This repo is the unofficial PyTorch reimplementation of "Peeking into the Future: Predicting Future Person Activities and Locations in Videos" in CVPR2019.

Original version: [next-prediction](https://github.com/google/next-prediction).

**Highlights:**
* We strongly follow the structure of [next-prediction](https://github.com/google/next-prediction) and change each line to PyTorch to make sure the model pipeline is consistent, except that we add `pip install wandb` to monitor the process.

* Pretrained weights are available 
[here](https://www.dropbox.com/scl/fi/vl9xvtpi1cg30p4640r7q/next-models.zip?rlkey=nvh02nokdajehqgziop71vo97&dl=0)

Put the `next-model/` under the current folder and run `bash scripts/run.sh --run_mode=test_single --runId=00`. You can get the following results.

```
-------------------------Test Single ----------------------
[['static', 13014], ['mov', 21166]]
loaded 34180 data points for test
total test samples:34180
restoring model...
load model from next-models/actev_single_model/model/00/best/save-best_64500.pt
saved output at single_model.traj.p.
performance:
0000_ade, 18.732775
0000_fde, 39.17251
0002_ade, 12.431181
0002_fde, 25.25748
0400_ade, 13.880731
0400_fde, 29.23549
0401_ade, 22.918083
0401_fde, 47.163574
0500_ade, 24.24398
0500_fde, 48.295353
act_ap, 0.19652319113923225
ade, 18.081436
fde, 37.212055
grid1_acc, 0.296875
grid2_acc, 0.39789719626168224
mov_ade, 20.305626
mov_fde, 42.167683
per_step_de_t0, 2.2559543
per_step_de_t1, 4.3897486
per_step_de_t10, 33.223103
per_step_de_t11, 37.212055
per_step_de_t2, 6.7923846
per_step_de_t3, 9.464146
per_step_de_t4, 12.347305
per_step_de_t5, 15.430407
per_step_de_t6, 18.692186
per_step_de_t7, 22.120876
per_step_de_t8, 25.676416
per_step_de_t9, 29.372652
static_ade, 14.464005
static_fde, 29.152206
traj_class_accuracy, 0.9410181392627267
traj_class_accuracy_0, 0.8987244505916705
traj_class_accuracy_1, 0.9670225833884531
0000_ade 0000_fde 0002_ade 0002_fde 0400_ade 0400_fde 0401_ade 0401_fde 0500_ade 0500_fde act_ap ade fde grid1_acc grid2_acc mov_ade mov_fde per_step_de_t0 per_step_de_t1 per_step_de_t10 per_step_de_t11 per_step_de_t2 per_step_de_t3 per_step_de_t4 per_step_de_t5 per_step_de_t6 per_step_de_t7 per_step_de_t8 per_step_de_t9 static_ade static_fde traj_class_accuracy traj_class_accuracy_0 traj_class_accuracy_1
18.732775 39.17251 12.431181 25.25748 13.880731 29.23549 22.918083 47.163574 24.24398 48.295353 0.19652319113923225 18.081436 37.212055 0.296875 0.39789719626168224 20.305626 42.167683 2.2559543 4.3897486 33.223103 37.212055 6.7923846 9.464146 12.347305 15.430407 18.692186 22.120876 25.676416 29.372652 14.464005 29.152206 0.9410181392627267 0.8987244505916705 0.9670225833884531
```
The raw code result is 

<table>
  <tr>
    <td>Activity mAP</td>
    <td>ADE</td>
    <td>FDE</td>
  </tr>
  <tr>
    <td>0.199</td>
    <td>17.979</td>
    <td>37.176</td>
  </tr>
</table>

There is one possible bug here we have not tested:
* the PyTorch version of tf.gather_nd

## Train/Test
Please use the raw scripts in [next-prediction](https://github.com/google/next-prediction) to train the model or use `bash scripts/run.sh ${your args}`. 

Note on `scripts/run.sh`: 
* args need to be assigned using the format `--{key}={0/1}` for booleans or `--{key}={value}` for others. 
* Use `--run_mode=train/test_single` to select whether training or testing

Welcome to contribute to this reimplementation and let me know if you find bugs!

If you find this code useful in your research then please cite

```
@InProceedings{Liang_2019_CVPR,
  author = {Liang, Junwei and Jiang, Lu and Carlos Niebles, Juan and Hauptmann, Alexander G. and Fei-Fei, Li},
  title = {Peeking Into the Future: Predicting Future Person Activities and Locations in Videos},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
```

### Updates
* **20 Nov 2023**: We have uploaded the [pretrained weights](https://www.dropbox.com/scl/fi/vl9xvtpi1cg30p4640r7q/next-models.zip?rlkey=nvh02nokdajehqgziop71vo97&dl=0) here. 
* **29 July 2022**: We have added `kaiming_normal_` for convolution weights, `trunc_normal_` for linear layers and `constant_` for biases. The performance boosts up and gets close to the official version 
* **4 Aug 2022**: We accelerated the data processing. It seems that Numpy is much faster than Pytorch. Now we can train the model within 2 hours.
* **6 Aug 2022**: Tested `embedding_lookup` which is correct. 
* **6 Aug 2022**: We followed the raw code and modified the dropout (important) and weight decay. We fixed one bug in `get_feed_dict`. Performance boosted.
* **23 Aug 2022 (important!)**: **Fixed bugs on `keep_prob` and focal attention**. We tested the model by `torch.nn.init.ones_` for weights and zeros initialisation for biases and printed each result. All results are comparable with the baseline.
