## 说明

	此为《人工智能入门实践》（肖波等著）7.4节代码，命名实体识别示例程序
	
	其中：
	
	(1) NER_train_test.py是将训练和测试合在一起的代码
	  执行时，先处理数据集，得到字典文件(ckpts/WordTag2id.pkl)，然后训练，生成模型文件(ckpts/ner.pkl)，然后测试，加载模型文件，进行测试，给出测试结果。顺序完成。
	
	(2) NERTrain.py、NERModel.py、NERTest.py是分开进行训练和测试的代码。代码内容和NER_train_test.py保持一致。
	  先执行NERTrain.py，得到字典文件(ckpts/WordTag2id.pkl)，然后训练，生成模型文件(ckpts/ner.pkl)。
	  再执行NERTest.py，进行字典文件(ckpts/WordTag2id.pkl)和模型文件(ckpts/ner.pkl)加载，测到测试结果。
	

## 执行程序前，安装必要的包：

	pip install -r requirements.txt
 
## 测试示例：

执行如下语句：

	python NERTest.py

结果如下：

		测试模型中...
		              precision    recall  f1-score   support
		
		      B-NAME     0.9901    0.8929    0.9390       112
		      E-NAME     1.0000    0.9554    0.9772       112
		           O     0.9683    0.9829    0.9755      5190
		      B-CONT     1.0000    1.0000    1.0000        28
		      M-CONT     0.9815    1.0000    0.9907        53
		      E-CONT     1.0000    1.0000    1.0000        28
		      B-RACE     1.0000    1.0000    1.0000        14
		      E-RACE     1.0000    1.0000    1.0000        14
		     B-TITLE     0.9237    0.9404    0.9320       772
		     M-TITLE     0.9345    0.9058    0.9199      1922
		     E-TITLE     0.9883    0.9845    0.9864       772
		       B-EDU     0.9730    0.9643    0.9686       112
		       M-EDU     0.9822    0.9274    0.9540       179
		       E-EDU     0.9909    0.9732    0.9820       112
		       B-ORG     0.9549    0.9566    0.9557       553
		       M-ORG     0.9672    0.9616    0.9644      4325
		       E-ORG     0.9271    0.9204    0.9238       553
		      M-NAME     0.9390    0.9390    0.9390        82
		       B-PRO     0.8529    0.8788    0.8657        33
		       M-PRO     0.7143    0.9559    0.8176        68
		       E-PRO     0.8611    0.9394    0.8986        33
		      S-NAME     0.0000    0.0000    0.0000         0
		       B-LOC     1.0000    0.8333    0.9091         6
		       M-LOC     1.0000    0.8571    0.9231        21
		       E-LOC     1.0000    0.8333    0.9091         6
		
		    accuracy                         0.9591     15100
		   macro avg     0.9180    0.9041    0.9092     15100
		weighted avg     0.9597    0.9591    0.9592     15100
		
		tags not in dataset: ['S-RACE', 'S-NAME', 'M-RACE', 'S-ORG']
