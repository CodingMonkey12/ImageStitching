# Panorama by using Image Stitching

This code is based on [Python+opencv 图像拼接](https://www.cnblogs.com/aaron-clark-aic/p/9994677.html), and has changed some codes to improve it.

Especially, it can use `more than two` pictures to make a panorama.

There is annotation in the file.

### Requirements

* Python 3.9
* Numpy 1.20.3
* OpenCV 4.5.5.64
* matplotlib 3.4.3
* tqdm 4.62.3

### To Run

You can find the file `ImageStitching.py` and run it.

### One of Examples

#### Inputs

<center class="half">
    <img src="https://s2.loli.net/2022/04/05/oFp9OaQvbXUlx1t.jpg" alt="A1" style="zoom: 10%;" />
    <img src="https://s2.loli.net/2022/04/05/ykO2QpcbifG1wnr.jpg" alt="A2" style="zoom:10%;" />
</center>
<center class="half">
    <img src="https://s2.loli.net/2022/04/05/DRCxWlzswYd7TrH.jpg" alt="A3" style="zoom:10%;" />
    <img src="https://s2.loli.net/2022/04/05/SgjvQklnCemLHAG.jpg" alt="A4" style="zoom:10%;" />
</center>



#### Output

![panorama](https://s2.loli.net/2022/04/05/riBHyVqcPoNEQTw.jpg)

Also, there are some pictures of the intermediate processing.

For example:

![panorama_processing_1](https://s2.loli.net/2022/04/05/T1Q8uHxFkhOp6qc.jpg)
