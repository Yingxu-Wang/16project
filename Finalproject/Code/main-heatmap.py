import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import pandas as pd
import seaborn as sns

class EnhancedInventoryModel:
    """增强版库存博弈模型"""
    def __init__(self, warehouse_data, shelf_data):
        self.warehouse = np.array(warehouse_data)
        self.shelf = np.array(shelf_data)
        self._validate_data()
        
        # 初始化博弈参数
        self.base_demand = 100
        self.price_elasticity = 0.5
        self.holding_cost = 0.2
        
    def _validate_data(self):
        """数据验证"""
        if len(self.warehouse) != len(self.shelf):
            raise ValueError("仓库和货架数据长度必须相同")
        if any(self.warehouse < 0) or any(self.shelf < 0):
            raise ValueError("库存量不能为负数")
            
    def calculate_heatmap_matrix(self):
        """生成热力图数据矩阵"""
        return np.vstack([self.warehouse, self.shelf])
    
    def analyze_strategy_impact(self):
        """策略影响分析"""
        strategy_impact = []
        for w, s in zip(self.warehouse, self.shelf):
            profit_w = (self.base_demand - self.price_elasticity*w)*w - self.holding_cost*w
            profit_s = (self.base_demand - self.price_elasticity*s)*s - self.holding_cost*s
            strategy_impact.append([profit_w, profit_s])
        return np.array(strategy_impact)
    
    def sensitivity_analysis(self, param_range, param_name='base_demand'):
        """参数敏感性分析"""
        results = []
        original_value = getattr(self, param_name)
        
        for value in param_range:
            setattr(self, param_name, value)
            matrix = self.calculate_heatmap_matrix()
            results.append(matrix.mean(axis=1))
            
        setattr(self, param_name, original_value)
        return pd.DataFrame(results, columns=['Warehouse', 'Shelf'], index=param_range)

class AdvancedHeatmapVisualizer:
    """高级热力图可视化器"""
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.cbar = None
        
    def create_basic_heatmap(self, data_matrix, title="库存热力图"):
        """创建基础库存热力图"""
        self.ax.clear()
        if self.cbar:
            self.cbar.remove()
            
        sns.heatmap(data_matrix, 
                   ax=self.ax,
                   annot=True,
                   fmt="d",
                   cmap="YlOrRd",
                   linewidths=0.5,
                   cbar_kws={'label': '库存量'})
        
        self.ax.set_title(title)
        self.ax.set_xlabel("时间周期")
        self.ax.set_ylabel("库存位置")
        self.ax.set_yticklabels(['仓库', '货架'])
        return self.fig
    
    def create_strategy_heatmap(self, strategy_matrix, title="策略影响分析"):
        """创建策略影响热力图"""
        self.ax.clear()
        if self.cbar:
            self.cbar.remove()
            
        sns.heatmap(strategy_matrix.T,
                   ax=self.ax,
                   annot=True,
                   fmt=".1f",
                   cmap="RdYlGn",
                   center=0,
                   cbar_kws={'label': '利润影响'})
        
        self.ax.set_title(title)
        self.ax.set_xlabel("时间周期")
        self.ax.set_ylabel("库存类型")
        self.ax.set_yticklabels(['仓库利润', '货架利润'])
        return self.fig
    
    def create_sensitivity_heatmap(self, df, title="参数敏感性分析"):
        """创建参数敏感性热力图"""
        self.ax.clear()
        if self.cbar:
            self.cbar.remove()
            
        sns.heatmap(df.T,
                   ax=self.ax,
                   annot=True,
                   fmt=".1f",
                   cmap="Blues",
                   cbar_kws={'label': '平均库存量'})
        
        self.ax.set_title(title)
        self.ax.set_xlabel("参数值")
        self.ax.set_ylabel("库存类型")
        return self.fig

class InventoryAnalysisApp(tk.Tk):
    """库存分析系统主界面"""
    def __init__(self):
        super().__init__()
        self.title("动态库存分析系统 v2.0")
        self.geometry("1200x800")
        
        self.model = None
        self.visualizer = AdvancedHeatmapVisualizer()
        self._create_widgets()
        self._setup_menu()
        
    def _create_widgets(self):
        """创建界面组件"""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="分析控制")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        ttk.Button(control_frame, text="加载数据", command=self.load_data).pack(pady=5)
        ttk.Button(control_frame, text="基础热力图", command=self.show_basic_heatmap).pack(pady=5)
        ttk.Button(control_frame, text="策略分析", command=self.show_strategy_heatmap).pack(pady=5)
        ttk.Button(control_frame, text="敏感性分析", command=self.show_sensitivity).pack(pady=5)
        
        # 可视化区域
        self.canvas_frame = ttk.Frame(main_frame)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 报告区域
        self.report_area = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD)
        self.report_area.pack(fill=tk.BOTH, expand=True)
        
    def _setup_menu(self):
        """创建菜单系统"""
        menu_bar = tk.Menu(self)
        
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="导出图像", command=self.export_image)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.quit)
        menu_bar.add_cascade(label="文件", menu=file_menu)
        
        self.config(menu=menu_bar)
        
    def load_data(self):
        """加载库存数据"""
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                df = pd.read_csv(file_path)
                warehouse = df['warehouse'].values
                shelf = df['shelf'].values
                self.model = EnhancedInventoryModel(warehouse, shelf)
                self.update_report("数据加载成功！\n" + df.describe().to_string())
            except Exception as e:
                self.update_report(f"数据加载失败：{str(e)}")
                
    def show_basic_heatmap(self):
        """显示基础热力图"""
        if self.model is None:
            return
            
        matrix = self.model.calculate_heatmap_matrix()
        self.visualizer.create_basic_heatmap(matrix)
        self.display_heatmap()
        
    def show_strategy_heatmap(self):
        """显示策略影响热力图"""
        if self.model is None:
            return
            
        strategy_matrix = self.model.analyze_strategy_impact()
        self.visualizer.create_strategy_heatmap(strategy_matrix)
        self.display_heatmap()
        
    def show_sensitivity(self):
        """显示参数敏感性分析"""
        if self.model is None:
            return
            
        param_range = np.linspace(80, 120, 5)
        df = self.model.sensitivity_analysis(param_range)
        self.visualizer.create_sensitivity_heatmap(df)
        self.display_heatmap()
        
    def display_heatmap(self):
        """在界面显示热力图"""
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
            
        canvas = FigureCanvasTkAgg(self.visualizer.fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def update_report(self, text):
        """更新分析报告"""
        self.report_area.delete(1.0, tk.END)
        self.report_area.insert(tk.END, text)
        
    def export_image(self):
        """导出当前图像"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")]
        )
        if file_path:
            self.visualizer.fig.savefig(file_path, dpi=300)

if __name__ == "__main__":
    app = InventoryAnalysisApp()
    app.mainloop()