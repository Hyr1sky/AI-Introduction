## 说明

	此为7.5节代码，AI自动作诗，代码参考陈云同学（github中的chenyuntc）代码.
	
	

## 执行程序前，安装必要的包：

  pip install -r requirements.txt
 
## 直接看藏头诗生成测试示例：

执行如下语句：

  python 7.5.5_poetryTest.py gen

由于fire库的支持，文件名7.5.5_poetryTest.py后的gen参数相当于执行该文件中的gen函数。结果如下：

  深山通海月，度水入江流。学道无人识，习家无事心。

执行如下语句：

  python 7.5.5_poetryTest.py gen --prefix_words=书山有路勤为径，学海无涯苦作舟。

执行结果为：
  
  深山不见无人识，度日不知何处是。学道不知何处所，习家不见心中见。

执行如下语句：

  python 7.5.5_poetryTest.py gen --prefix_words=欲穷千里目，更上一层楼。 -start_words=北京邮电大学

执行结果如下：

	北阙三千里，京州万里游。邮亭连北阙，电气入秦楼。大道多人世，学人皆有心。

代码使用了poetry.npz语料数据文件和poetry_model.pth模型文件

## 模型训练，得到poetry_model.pth模型文件：

训练命令：

  python poetryTrain.py

代码使用了poetry.npz语料数据文件
  
## 语料数据文件poetry.npz可再次生成:

执行代码：

  python poetryDataset.py