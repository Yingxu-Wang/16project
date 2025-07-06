import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class GoodsDetector:
    """货物识别模块"""
    def __init__(self):
        # 模拟摄像头识别结果（实际应替换为实际识别逻辑）
        self.recognition_history = []
    
    def detect_goods(self):
        """模拟实时货物识别（返回仓库和货架数量）"""
        # 实际应用中应连接摄像头识别系统
        warehouse = np.random.randint(0, 5)  # 模拟仓库识别结果
        shelf = np.random.randint(0, 8)      # 模拟货架识别结果
        self.recognition_history.append((warehouse, shelf))
        return warehouse, shelf

class InventoryAnalyzer:
    """库存分析系统"""
    def __init__(self, detector):
        self.detector = detector
        self.analysis_history = []
        
    def collect_data(self, days=3):
        """收集近期数据"""
        # 获取最近N次识别结果（模拟多日数据）
        recent_data = self.detector.recognition_history[-days:]
        return recent_data if recent_data else [(0,0)]*days
    
    def generate_heatmap(self, data):
        """生成热力图分析"""
        df = pd.DataFrame(data, columns=['Warehouse', 'Shelf'])
        plt.figure(figsize=(10, 4))
        sns.heatmap(df.T, annot=True, fmt="d", cmap="YlOrRd")
        plt.title("Recent Inventory Heatmap")
        plt.xlabel("Days")
        plt.ylabel("Storage Type")
        plt.show()
    
    def analyze_trend(self, data):
        """分析库存趋势"""
        warehouse = [x[0] for x in data]
        shelf = [x[1] for x in data]
        
        # 计算关键指标
        avg_warehouse = np.mean(warehouse)
        avg_shelf = np.mean(shelf)
        stockout_rate = sum(1 for w,s in data if w==0 or s==0)/len(data)
        
        return {
            'avg_warehouse': avg_warehouse,
            'avg_shelf': avg_shelf,
            'stockout_rate': stockout_rate,
            'last_ratio': shelf[-1]/(warehouse[-1]+1e-6)  # 防止除零
        }
    
    def generate_advice(self, analysis):
        """生成动态建议"""
        advice = []
        
        # 根据货架占比建议
        if analysis['last_ratio'] > 0.7:
            advice.append("货架库存占比过高，建议调整陈列布局")
        elif analysis['last_ratio'] < 0.3:
            advice.append("货架库存占比过低，建议增加补货频次")
            
        # 根据缺货率建议
        if analysis['stockout_rate'] > 0.3:
            advice.append("高频缺货，需检查供应链流程")
            
        # 根据平均库存建议
        if analysis['avg_warehouse'] < 2:
            advice.append("仓库安全库存不足，建议增加采购量")
            
        return advice if advice else ["库存状态健康，保持当前策略"]

# 使用示例
if __name__ == "__main__":
    # 初始化系统
    detector = GoodsDetector()
    analyzer = InventoryAnalyzer(detector)
    
    # 模拟3天数据采集
    print("=== 实时货物识别 ===")
    for _ in range(3):
        wh, sh = detector.detect_goods()
        print(f"识别结果：仓库={wh}件，货架={sh}件")
    
    # 获取分析数据
    recent_data = analyzer.collect_data()
    
    # 生成热力图
    print("\n=== 热力图分析 ===")
    analyzer.generate_heatmap(recent_data)
    
    # 生成建议报告
    analysis = analyzer.analyze_trend(recent_data)
    advice = analyzer.generate_advice(analysis)
    
    print("\n=== 智能建议 ===")
    print(f"基于最近{len(recent_data)}天数据分析：")
    for item in advice:
        print(f"• {item}")
    print(f"当前货架占比：{analysis['last_ratio']:.0%}")
    print(f"平均缺货率：{analysis['stockout_rate']:.0%}")