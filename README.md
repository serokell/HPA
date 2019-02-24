# Human Protein Atlas image classification challenge
#### 67th place solution [silver]  

An example of data sample (visualisation has been adopted from [this kernel](https://www.kaggle.com/allunia/protein-atlas-exploration-and-baseline)):
![Data Sample](https://habrastorage.org/webt/f7/np/nh/f7npnh-6isv4xsnecbzf5xgdkzo.png)

ResNet50 with extended 4-channel input was used along with focal loss and over-under sumpling technique.
On the hist below: 1. original distribution, 2. over-under sumpled).  
![Hist](https://habrastorage.org/webt/sh/kz/5d/shkz5dnu3eh8usjtbjjyjkjgb7e.png)

Learning procedure has been performed with `warm restarts` for last 32 epochs, followed by `lr-decay on plato` police for first 100. Learning curves along with F-score for eachh label are depicted on the plots below:
![Plot](https://habrastorage.org/webt/d9/rf/dv/d9rfdvtrbde9lity1inlih98ynq.png)
