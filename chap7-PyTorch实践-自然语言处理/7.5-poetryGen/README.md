## ˵��

	��Ϊ7.5�ڴ��룬AI�Զ���ʫ������ο�����ͬѧ��github�е�chenyuntc������.
	
	

## ִ�г���ǰ����װ��Ҫ�İ���

  pip install -r requirements.txt
 
## ֱ�ӿ���ͷʫ���ɲ���ʾ����

ִ��������䣺

  python 7.5.5_poetryTest.py gen

����fire���֧�֣��ļ���7.5.5_poetryTest.py���gen�����൱��ִ�и��ļ��е�gen������������£�

  ��ɽͨ���£���ˮ�뽭����ѧ������ʶ��ϰ�������ġ�

ִ��������䣺

  python 7.5.5_poetryTest.py gen --prefix_words=��ɽ��·��Ϊ����ѧ�����Ŀ����ۡ�

ִ�н��Ϊ��
  
  ��ɽ��������ʶ�����ղ�֪�δ��ǡ�ѧ����֪�δ�����ϰ�Ҳ������м���

ִ��������䣺

  python 7.5.5_poetryTest.py gen --prefix_words=����ǧ��Ŀ������һ��¥�� -start_words=�����ʵ��ѧ

ִ�н�����£�

	������ǧ����������Ρ���ͤ�����ڣ���������¥�������������ѧ�˽����ġ�

����ʹ����poetry.npz���������ļ���poetry_model.pthģ���ļ�

## ģ��ѵ�����õ�poetry_model.pthģ���ļ���

ѵ�����

  python poetryTrain.py

����ʹ����poetry.npz���������ļ�
  
## ���������ļ�poetry.npz���ٴ�����:

ִ�д��룺

  python poetryDataset.py