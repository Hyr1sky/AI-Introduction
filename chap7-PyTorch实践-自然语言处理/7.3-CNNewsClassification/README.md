## 说明

	此为《人工智能入门实践》（肖波等著）7.3节代码，中文文本分类示例程序
	
	其中：
	
	(1) trainTextCNN.py是训练TextCNN模型
	  执行时，先处理数据集（data目录中的train.txt, dev.txt），
	  然后训练，生成模型文件(checkpoint/textcnn_model.0.xxx)。
	  训练过程需要embedding_SougouNews.npz文件。
	  
	(2) testTextCNN.py是TextCNN模型测试的代码（需要用到trainTextCNN.py中的部分模块）。
	
	(3) trainRbt3.py是训练roberta3模型,也需要用到trainTextCNN.py中的部分模块。
  
	(4) testRbt3.py是roberta3模型测试的代码,也需要用到trainTextCNN.py中的部分模块。
  
  
## 执行程序前，安装必要的包：

	pip install -r requirements.txt
 

## 测试示例：

执行如下语句：

	python testRbt3.py

结果如下：

		Loading dataset data/test.txt ...
		Test Loss:  0.21,  Test Acc: 93.50%
		Precision, Recall and F1-Score...
		              precision    recall  f1-score   support
		
		          体育     0.9759    0.9900    0.9829      2000
		          娱乐     0.9395    0.9625    0.9509      2000
		          家居     0.9402    0.9200    0.9300      2000
		          教育     0.9246    0.9560    0.9400      2000
		          时政     0.8867    0.9035    0.8950      2000
		          游戏     0.9704    0.9350    0.9524      2000
		          社会     0.9259    0.9190    0.9225      2000
		          科技     0.9328    0.9370    0.9349      2000
		          股票     0.9308    0.9075    0.9190      2000
		          财经     0.9251    0.9195    0.9223      2000
		
		    accuracy                         0.9350     20000
		   macro avg     0.9352    0.9350    0.9350     20000
		weighted avg     0.9352    0.9350    0.9350     20000
		
		Confusion Matrix...
		[[1980   10    2    1    3    0    2    1    1    0]
		 [  16 1925   13    7   13    8    6    7    0    5]
		 [   4   25 1840   23   25    7   17   24   19   16]
		 [   6    6   10 1912   13    2   36   12    3    0]
		 [   8   22   14   44 1807    5   28   35   23   14]
		 [   1   20   19   13   13 1870   26   23   10    5]
		 [   4   22   17   27   57    4 1838   24    2    5]
		 [   2    5   16   23   29   23   13 1874    9    6]
		 [   3    2   12   11   50    5    2    2 1815   98]
		 [   5   12   14    7   28    3   17    7   68 1839]]
		 

## embedding_SougouNews.npz文件是如何得到的？

    执行embedding.py文件，读取data目录下的train.txt文件和embedding目录下的sgns.sogou.char文件，处理后，得到该文件。
    
注意：embedding目录下的sgns.sogou.char文件需要从[https://gitee.com/flycity/ai_tutorial_book_datafile](https://gitee.com/flycity/ai_tutorial_book_datafile)下载所有的sgns.sogou.char.z*文件解压缩得到。