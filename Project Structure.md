## Project Structure

------

└─bacteria  
    ├─code				tensorflow分类程序  
    │             
    ├─code_sjh		pytorch分类程序（主要程序）  
    │  │    
    │  ├─bin   
    │  │  │  bectrtia_Sun_AST.py  		卷积分类脚本（待重命名）  
    │  │  │  CVAE_exosome.py  			条件变分自编码器脚本（待重命名）  
    │  │  │  preprocess_data_20211123.py  	预处理过程可视化脚本  
    │  │  │  preprocess_data_test.py  
    │  │  │  radar.py  								多通道数据分类脚本  
    │  │  │  __init__.py  
    │  │  
    │  ├─checkpoints			存放训练后网络权重  
    │  │            
    │  ├─config		      	存放设置文件（TODO  
    │  │            
    │  ├─GUI		       	存放GUI文件（TODO  
    │  │                
    │  ├─ML		         	存放机器学习脚本（TODO  
    │  │  │    
    │  │  └─PCA  
    │  │            
    │  ├─models		      	存放网络模型  
    │  │  │  BasicModule.py	        	定义本项目模型的一些基本功能（父类）  
    │  │  │  __init__.py  
    │  │  │    
    │  │  ├─AutoEncoders	          	存放自编码器  
    │  │  │  │  AutoEncoder.py	    	自编码器  
    │  │  │  │  AutoEncoderTest.py	自编码器测试脚本（待转移  
    │  │  │  │  CVAE.py	            	条件变分自编码器  
    │  │  │  │  VAE.py		          	变分自编码器  
    │  │  │  │  __init__.py  
    │  │  │             
    │  │  ├─CNN		                存放卷积网络  
    │  │  │  │  AlexNet.py		Alexnet （只有一种  
    │  │  │  │  GoogleLeNet.py	    	（未实现）  
    │  │  │  │  ResNet.py	          	Resnet18/34/50/101/150  
    │  │  │  │  SVM.py	            	纯线性网络（应该命名为Dense，但早期偷懒了没改  
    │  │  │  │  VGG.py	            	（未实现）  
    │  │  │  │  __init__.py  
    │  │  │           
    │  │  ├─Criteons		            存放损失函数  
    │  │  │  │  CostSensitive.py	  	成本敏感度（未实现）  
    │  │  │  │  CrossEntropy.py	    	交叉熵（BCE）  
    │  │  │  │  RelativeEntropy.py	    相对熵/KL散度（VAE使用）  
    │  │  │  │  __init__.py  
    │  │  │          
    │  │  └─Parts		                存放网络组件  
    │  │     │  decoders.py	        	解码器  
    │  │     │  encoders.py	        	编码器  
    │  │     │  layers.py	          	网络层  
    │  │     │  __init__.py  
    │  │            
    │  └─utils
    │     │  Classifier.py	        	分类结果输出模块  
    │     │  iterator.py	         	网络训练模块  
    │     │  Process.py	            	数据预处理模块  
    │     │  RamanData.py	          	数据集读取分割模块  
    │     │  __init__.py  
    │     │   
    │     ├─Process_utils	          	存放数据预处理工具  
    │     │  │  AuRemoval.py	      	背景信号去除（待改名）  
    │     │  │  baseline_remove.py	    基线去除  
    │     │  │  __init__.py  
    │     │            
    │     └─Validation	            	存放验证工具  
    │        │  mpl_utils.py	      	matplotlib可视化工具  
    │        │  validation.py	      	性能评估模块  
    │        │  visdom_utils.py	    	visdom可视化工具  
    │        │  __init__.py  
    │        │    
    │        └─hooks			存放hooks  
    │           │  backward_hooks.py  
    │           │  forward_hooks.py  
    │           │  forward_pre_hooks.py  
    │           │  __init__.py  
    │            
    ├─data 			存放实验数据  
    │           
    └─result 			存放实验结果  