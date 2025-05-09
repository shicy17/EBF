# EBF: An Event-based Bilateral Filter for Effective Neuromorphic Vision Sensor Denoising

# Abstract
Neuromorphic Vision Sensors (NVS) have raised increasing attention due to their sparsity, low latency, and high dynamic range. However, they suffer from the background activity noise which causes unnecessary computational waste. Existing learning-based denoising methods usually achieve better performance than rule-based methods but require larger computational and storage resources. To make rule-based filters as competitive as learning-based filters, this paper proposes a novel filter, namely the Event-based Bilateral Filter (EBF) that utilizes both spatiotemporal and polarity information. EBF first assigns two types of weights to each nearest neighborhood pixel based on the temporal and polarity information of the event to be classified. Next, EBF multiplies and accumulates the weights to get a correlation score, which is then compared with a threshold to predict the label of the event. We evaluate the proposed methods on three neuromorphic datasets, including both simulated data and real-world data. EBF achieves the best denoising accuracy compared with other rule-based and learning-based denoising methods across different noise levels.

# :point_right: Citation
Citations are welcome.
```
@ARTICLE{10994567,
  author={Guo, Shasha and Shi, Chenyang and Wang, Lei and Jin, Jing and Lu, Yuliang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={EBF: An Event-based Bilateral Filter for Effective Neuromorphic Vision Sensor Denoising}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Noise reduction;Information filters;Noise;Correlation;Accuracy;Spatiotemporal phenomena;Cameras;Mathematical models;Artificial neural networks;Training;Neuromorphic Vision Sensor;denoising;event;bilateral filter},
  doi={10.1109/TCSVT.2025.3568604}}

```

# :star: BUAA_Campus Denoising Dataset
[Google drive: BUAA_Campus Denoising Dataset](https://drive.google.com/drive/folders/1NiswzR7yJ9z_nxOyc7dXW76b0hXZdViz?usp=sharing). For academic use only. This dataset is a part of our article "Event-Based Bilateral Filter for Effective Neuromorphic Vision Sensor Denoising". 

The dataset was captured using an iPhone and transformed into event sequences with a resolution of $346\times260$ through the v2e method. This transformation ensures that the event sequences are free from noise. The dataset comprises two sequences recorded on the campus of Beihang University. During these recordings, the photographer was seated on an electric bike, incorporating significant camera movements to enhance the dynamic capture of the scene. The scenes themselves are richly textured and bustling with dynamic objects such as pedestrians and vehicles, posing significant challenges for denoising tasks.

For a fari comparison, we provide the original event sequences and sequences with noise at level of 1Hz/pixel, 3Hz/pixel and 5Hz/pixel, respectively.


# :dizzy: Demonstartion
![Campus1](https://github.com/shicy17/BUAA_campus/blob/main/Demonstration/Campus1.gif?raw=true "Campus1") ![Campus2](https://github.com/shicy17/BUAA_campus/blob/main/Demonstration/Campus2.gif?raw=true "Campus2")


# :clap: Acknowledgement
We gratefully acknowledge Professor Tobi Delbruck for implementing EBF (as [AgePolarityDenoiser](https://github.com/SensorsINI/jaer/blob/master/src/net/sf/jaer/eventprocessing/filter/AgePolarityDenoiser.java)) and PFD (as [STCF with polarity](https://github.com/SensorsINI/jaer/blob/master/src/net/sf/jaer/eventprocessing/filter/SpatioTemporalCorrelationFilter.java)) algorithms in the [jaer](https://github.com/SensorsINI/jaer) project.

Note that since the Campus dataset includes a vast number of events, we utilized the first 3 million events from each sequence for our experiments.

# Reference

[Hu, Yuhuang, Shih-Chii Liu, and Tobi Delbruck. "v2e: From video frames to realistic DVS events." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.](http://arxiv.org/abs/2006.07722)



