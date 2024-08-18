my_project/
│
├── data/
│   ├── raw/                # 原始数据
│   ├── processed/          # 处理后的数据
│   ├── external/           # 外部数据
│   │   ├── basemap-province.geojson  # 省级行政区边界数据
│   │   ├── basemap-city.geojson      # 市级行政区边界数据
│   │   └── outline.geojson           # 外轮廓数据
│   └── metadata.yaml       # 数据集元数据文件
│
├── notebooks/              # Jupyter 笔记本
│   └── exploratory.ipynb   # 数据探索笔记本
│
├── src/
│   ├── __init__.py
│   ├── data/               # 数据处理脚本
│   │   └── generate_outline.py  # 生成外轮廓的脚本
│   ├── features/           # 特征工程脚本
│   ├── models/             # 模型定义与训练脚本
│   └── visualization/      # 可视化脚本
│       └── plot_basemap.py  # 绘制地图的脚本
│
├── models/                 # 训练好的模型
│   └── ...
│
├── reports/                # 报告与结果
│   └── figures/            # 可视化图表
│       └── basemap.png     # 最终绘制的地图
│
├── .gitignore              # Git 忽略文件配置
├── README.md               # 项目说明文档
├── requirements.txt        # Python 依赖包列表
├── environment.yml         # Conda 环境配置文件
└── LICENSE                 # 许可证文件
