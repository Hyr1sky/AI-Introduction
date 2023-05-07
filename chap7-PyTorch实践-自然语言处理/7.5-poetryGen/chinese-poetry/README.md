## 说明

	json目录下的所有文件来自于：
	
	https://github.com/chinese-poetry/chinese-poetry
	
	更为详细的古诗词信息请访问该网址。在此向作者表示感谢！

  simplified目录下的文件可以通过执行当前目录下的 toSimplified.py 文件来生成。
  
  实际训练代码中只用到 simplified 目录下的json文件。
  
## simplified 目录下的所有文件数据形式

*poet.tang.[0-99000].json*

*poet.song.[0-57000].json*

每个 JSON 文件有1000条诗. 示例如下：

	[
	  {
	    "id": "08e41396-2809-423d-9bbc-1e6fb24c0ca1",
	    "title": "日诗",
	    "paragraphs": "欲出未出光辣达，千山万山如火发。须臾走向天上来，逐却残星赶却月。",
	    "author": "宋太祖"
	  },
	  
	  ...
	]


## json目录下的所有文件数据形式

*poet.tang.[0-99000].json*

*poet.song.[0-57000].json*

每个 JSON 文件有1000条诗.

```text
[
  {
    "strains": [
      "平平平仄仄，平仄仄平平。",
      "仄仄平平仄，平平仄仄平。",
      "平平平仄仄，平仄仄平平。",
      "平仄仄平仄，平平仄仄平。"
    ],
    "author": "太宗皇帝",
    "paragraphs": [
      "秦川雄帝宅，函谷壯皇居。",
      "綺殿千尋起，離宮百雉餘。",
      "連甍遙接漢，飛觀迥凌虛。",
      "雲日隱層闕，風煙出綺疎。"
    ],
    "title": "帝京篇十首 一"
  }
]
```

注意: 为了举例方便， 省略了剩下999篇诗.


