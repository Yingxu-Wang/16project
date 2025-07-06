Final-project/  
├── code/                     # 核心代码层，实现系统功能逻辑与硬件交互  
│   ├── inventory_dashboard/  # 库存看板模块，负责前端展示与数据可视化  
│   │   ├── static/           # 静态资源目录，支撑页面渲染与交互  
│   │   │   ├── css/          # 样式文件，定义页面布局与视觉风格  
│   │   │   │   └── style.css # 核心样式表，控制看板组件排版、配色方案  
│   │   │   ├── images/       # 静态图片资源，用于商品展示与背景装饰  
│   │   │   │   ├── bg.png    # 页面背景图，提供统一视觉基调  
│   │   │   │   ├── chips1.jpg # 薯片1商品图，用于库存列表展示  
│   │   │   │   ├── chips2.jpg # 薯片2商品图，用于库存列表展示  
│   │   │   │   └── peanuts.jpg # 花生商品图，用于库存列表展示  
│   │   │   ├── js/           # 前端交互脚本，实现动态数据加载与用户操作响应  
│   │   │   │   └── app.js    # 主逻辑脚本，处理页面按钮点击、数据接口调用  
│   │   │   └── analysis.png  # 热力图/趋势图示例图，用于前端可视化组件演示  
│   │   ├── templates/        # HTML模板目录，定义页面结构与动态数据渲染点  
│   │   │   └── index.html    # 看板首页模板，包含导航栏、库存数据表格、可视化图表区域  
│   │   ├── app.py            # 后端主程序，基于Flask框架实现路由与数据接口  
│   │   ├── detect_inventory.py # 库存检测脚本，调用摄像头识别商品并校验库存数量  
│   │   ├── history.csv       # 库存历史记录文件，存储每日库存变动数据（如补货、出库记录）  
│   │   └── inventory.json    # 当前库存数据文件，存储商品SKU、数量、位置等结构化信息  
│   ├── lift.ino              # Arduino硬件控制代码，实现货架升降与机械臂动作控制  
│   ├── main-function.py      # 系统主程序，整合硬件控制、视觉识别与业务逻辑  
│   ├── main-heatmap.py       # 热力图生成模块，基于库存数据绘制可视化图表  
│   ├── main-suggestion.py    # 智能建议模块，分析库存趋势并生成补货策略  
│   └── xiaonuo.py            # 语音助手模块，实现"小诺"语音交互与配送状态播报  
├── data/                     # 数据层，存放原始数据、模型训练数据与预处理脚本  
│   └── train12/              # 模型训练数据集目录（含图片与标注）  
│       ├── images/           # 训练图片文件夹  
│       │   ├── chips1.jpg    # 薯片1训练样本  
│       │   ├── chips2.jpg    # 薯片2训练样本  
│       │   └── peanuts.jpg   # 花生训练样本  
│       ├── labels/           # 标注文件文件夹（如YOLO格式txt文件）  
│       │   ├── chips1.txt    # 薯片1标注信息  
│       │   ├── chips2.txt    # 薯片2标注信息  
│       │   └── peanuts.txt   # 花生标注信息  
│       ├── best.pt           # 训练最优模型权重文件（用于目标检测）  
│       └── last.pt           # 最后一次训练模型权重文件  
├── model/                    # 模型层，存储算法模型脚本与训练逻辑  
│   └── Dynamic-zero-sum-game-model.py # 动态零和博弈模型，用于库存策略优化  
│       # 功能：定义博弈论模型结构，分析仓库-货架库存分配策略，输出最优补货方案  
└── Readme                    # 项目说明文档（需补充内容）  
 

代码介绍
		main-function是最主要的代码，可以运行的（我们简称主代码）
		inventory_dashboard里面是前端网页的代码
		lift是Arduino 的代码负责被主代码调用,底层硬件控制脚本，负责货架升降、机械臂抓取等物理动作执行。
		main-heatmap是生成热力图的代码负责被前端调用
		main-suggestion生成建议的代码负责被主代码调用前端
		xiaonuo是小诺智能语音的代码负责被主代码调用
		static/css/style.css定义页面布局与样式（如数据表格、图表区域）；
		static/js/app.js实现前端交互逻辑（如按钮点击加载数据、图表动态渲染）；
		templates/index.html为看板首页模板，通过 Jinja2 渲染动态数据（如库存数量、热力图）。


环境配置
		Arduino IDE 1.8.19+ 
		Python 版本：3.8+（需支持dataclasses、typing等特性）
		Web 浏览器：Chrome 80+ / Firefox 75+（支持 HTML5、CSS3 及 JavaScript ES6）
		后端服务：Flask 开发服务器（inventory_dashboard/app.py）
		百度API


