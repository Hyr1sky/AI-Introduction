## 说明

	此为《人工智能入门实践》（肖波等著）6.4节代码，实例分割示例程序
	
	其中：
	（1）InstanceSeg_mobileNet.py：应用预训练模型进行实例分隔测试代码，即使用已经训练好的模型直接测试。
	改代码与前一节代码基本相同，修改地方仅用两处，具体参见教材。
	程序运行需要字体文件arial.ttf的支持。代码中的图像文件可以修改为自己的文件。

	程序执行后，目标被标注后的结果图像存储在新生成的result.png文件中。
  
	（2）showDataset.py：用于显示数据集中的图像和标注信息。

	（3）InstanceSeg_trainPenn.py：使用PennFudanPed数据集，重新训练模型。
	该训练过程需要PennFudanPed数据集。该数据集来源于：
	https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
	本目录中也已经包含，需要解压缩到本目录中，目录结构如下：
	PennFudanPed/
	  PedMasks/
	    FudanPed00001_mask.png
	    FudanPed00002_mask.png
	    FudanPed00003_mask.png
	    FudanPed00004_mask.png
	    ...
	  PNGImages/
	    FudanPed00001.png
	  FudanPed00002.png
	    FudanPed00003.png
	    FudanPed00004.png
	    ...
	其中，PNGImages目录为原始图像，PedMasks目录为对应的标注图像。需要注意的是，标注图像实际上仅仅对原始图像中的实例进行了每个像素的掩码标注，属于同一实例的像素对应相同的掩码编码。
	
	（4）InstanceSeg_testPenn.py：使用重新训练的模型进行测试。
	
## 执行程序前，安装必要的包：

	pip install -r requirements.txt
 

