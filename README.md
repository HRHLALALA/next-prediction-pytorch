# next-prediction.pytorch
 
This repo is the unofficial PyTorch reimplementation of [next-prediction](https://github.com/google/next-prediction).

We strongly follow the structure of [next-prediction](https://github.com/google/next-prediction) and change each line to PyTorch to make sure the model pipeline is consistent, except that we add `pip install wandb` to monitor the process.

We have trained and tested the model. Here is the results:

```
performance:
0000_ade, 19.471685
0000_fde, 40.906437
0002_ade, 12.635235
0002_fde, 25.589851
0400_ade, 13.957309
0400_fde, 29.448845
0401_ade, 23.380247
0401_fde, 48.150265
0500_ade, 23.349108
0500_fde, 47.016354
act_ap, 0.19804641877328344
ade, 18.419632
fde, 37.957672
grid1_acc, 0.29634929906542057
grid2_acc, 0.398160046728972
mov_ade, 20.651001
mov_fde, 42.804962
static_ade, 14.790522
static_fde, 30.074034
traj_class_accuracy, 0.9378876535985957
traj_class_accuracy_0, 0.8856615952051636
traj_class_accuracy_1, 0.9699990550883493
0000_ade 0000_fde 0002_ade 0002_fde 0400_ade 0400_fde 0401_ade 0401_fde 0500_ade 0500_fde act_ap ade fde grid1_acc grid2_acc mov_ade mov_fde static_ade static_fde traj_class_accuracy traj_class_accuracy_0 traj_class_accuracy_1
19.471685 40.906437 12.635235 25.589851 13.957309 29.448845 23.380247 48.150265 23.349108 47.016354 0.19804641877328344 18.419632 37.957672 0.29634929906542057 0.398160046728972 20.651001 42.804962 14.790522 30.074034 0.9378876535985957 0.8856615952051636 0.9699990550883493
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

There is one possible bugs here we have not tested:
* the pytorch version of tf.gather_nd

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
* **29 July 2022**: We have added `kaiming_normal_` for convolution weights, `trunc_normal_` for linear layers and `constant_` for biases. The performance boosts up and gets close to the official version 
* **4 Aug 2022**: We accelerated the data processing. It seems that Numpy is much faster than Pytorch. Now we can train the model within 2 hours.
* **6 Aug 2022**: Tested `embedding_lookup` which is correct. 
* **7 Aug 2022**: We followed raw code and modified the dropout and weight decay. We fixed one bug in `get_feed_dict`. Performance boosted.
 * The position of dropout is very important
