从0开始运行Bert的流程：
步骤一：运行本文件夹下的ExtractEmbedding.py文件，从data.csv中生成tmp_embedding文件夹，每个测试数据集样本生成一个index对应的文件夹。
备注1：Todo：由于部分原因导致的Bert中的生成token无法使用，所以出此下策(好像是环境问题)
备注2：现在的git中已经有tmp_embedding文件夹所以无需运行。可以忽略步骤一。
步骤二：然后运行FHE-BERT-Tiny.py运行主程序，进行测试。