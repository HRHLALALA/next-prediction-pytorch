# next-prediction.pytorch
 
This repo is the unofficial PyTorch reimplementation of [next-prediction](https://github.com/google/next-prediction).

We strongly follow the structure of [next-prediction](https://github.com/google/next-prediction) and change each line to PyTorch to make sure the model pipeline is consistent, except that we add `pip install wandb` to monitor the process.

We have trained and tested the model. Here is the results:

```
performance:
0000_ade, 19.070852
0000_fde, 40.016636
0002_ade, 12.5598135
0002_fde, 25.465317
0400_ade, 13.849079
0400_fde, 29.020937
0401_ade, 22.955137
0401_fde, 47.329758
0500_ade, 24.321432
0500_fde, 48.572792
act_ap, 0.1946390249960858
ade, 18.19456
fde, 37.477123
grid1_acc, 0.2964077102803738
grid2_acc, 0.39742990654205607
mov_ade, 20.38324
mov_fde, 42.333584
per_step_de_t0, 2.2617185
per_step_de_t1, 4.4127126
per_step_de_t10, 33.43781
per_step_de_t11, 37.477123
per_step_de_t2, 6.835699
per_step_de_t3, 9.523781
per_step_de_t4, 12.420231
per_step_de_t5, 15.527474
per_step_de_t6, 18.804314
per_step_de_t7, 22.252695
per_step_de_t8, 25.82756
per_step_de_t9, 29.55357
static_ade, 14.634879
static_fde, 29.578573
traj_class_accuracy, 0.9385898186073728
traj_class_accuracy_0, 0.8952666359305363
traj_class_accuracy_1, 0.9652272512520079
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
* **6 Aug 2022**: We followed raw code and modified the dropout (important) and weight decay. We fixed one bug in `get_feed_dict`. Performance boosted.
* **23 Aug 2022 (important!)**: **Fixed bugs on `keep_prob` and focal attention**. We tested the model by ones initialisation for weights and zeros initialisation for biases and print each result. All results are comparable with baseline.
