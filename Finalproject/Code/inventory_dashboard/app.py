from flask import Flask, render_template, jsonify, request
import threading
import time
import json
import os

# Matplotlib non-interactive backend
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 新版 OpenAI SDK（兼容 DeepSeek）
from openai import OpenAI

# —— DeepSeek 客户端配置 ——
ds_api_key = os.getenv("DEEPSEEK_API_KEY")  # 请先在环境变量中设置
client = OpenAI(
    api_key=ds_api_key,
    base_url="https://api.deepseek.com/v1"
)

app = Flask(__name__)

# 全局库存结构，初始化为 0
inventory = {"chips1": 0, "chips2": 0, "peanuts": 0}
MAX_STOCK = 5

# —— 引入自定义热力图模型与可视化 ——
import numpy as np


class InventoryGameModel:
    def __init__(self, warehouse_data, shelf_data):
        self.warehouse_data = warehouse_data
        self.shelf_data = shelf_data
        self.days = len(warehouse_data)

    def calculate_sales_coefficient(self):
        coeffs = []
        for i in range(1, self.days):
            shelf_change = self.shelf_data[i] - self.shelf_data[i - 1]
            warehouse_change = self.warehouse_data[i - 1] - self.warehouse_data[i]
            coeff = shelf_change / warehouse_change if warehouse_change != 0 else 0
            coeffs.append(coeff)
        expected = float(np.mean(coeffs)) if coeffs else 0.0
        return coeffs, expected


class InventoryVisualizer:
    def __init__(self, model):
        self.model = model

    def create_heatmap(self, fig=None):
        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            ax = fig.subplots()
        matrix = np.array([self.model.warehouse_data, self.model.shelf_data])
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(np.arange(self.model.days))
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f'Day {i + 1}' for i in range(self.model.days)])
        ax.set_yticklabels(['Warehouse', 'Shelf'])
        for i in range(2):
            for j in range(self.model.days):
                ax.text(j, i, matrix[i, j], ha='center', va='center')
        ax.set_title('Inventory Heatmap')
        fig.colorbar(im, ax=ax)
        return fig

    def generate_report(self):
        _, exp = self.model.calculate_sales_coefficient()
        return f"预计销售系数: {exp:.2f}\n"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/inventory")
def get_inventory():
    data = [{"name": n, "count": c, "missing": max(0, MAX_STOCK - c)}
            for n, c in inventory.items()]
    return jsonify(data)


@app.route("/api/control", methods=["POST"])
def control_robot():
    action = request.json.get("action")
    return jsonify({"status": "ok", "action": action})


@app.route("/api/analysis")
def analysis():
    # 示例热力图数据（按需替换为真实历史）
    warehouse_data = [1, 1, 0, 4]
    shelf_data = [2, 3, 2, 5]
    model = InventoryGameModel(warehouse_data, shelf_data)
    viz = InventoryVisualizer(model)

    # 1) 生成并保存热力图
    fig = viz.create_heatmap()
    chart_path = os.path.join(os.path.dirname(__file__), 'static', 'analysis.png')
    fig.savefig(chart_path, bbox_inches='tight')
    plt.close(fig)

    # 2) 调用 DeepSeek 生成分析文字
    system_msg = "你是库存分析师。"
    user_msg = (
            "请根据以下消耗数据简要分析并给出补货建议：\n" +
            "\n".join(f"{k}: {v}" for k, v in zip(['chips1 薯惑（爽口青瓜味）', 'chips2 薯惑（浓情烤肉味）', 'peanuts 黄飞红麻辣花生'],
                                                  [MAX_STOCK - x for x in warehouse_data]))
    )
    try:
        resp = client.chat.completions.create(
            model='deepseek-chat',
            messages=[
                {'role': 'system', 'content': system_msg},
                {'role': 'user', 'content': user_msg}
            ],
            stream=False
        )
        analysis_text = resp.choices[0].message.content.strip()
    except Exception as e:
        import traceback;
        traceback.print_exc()
        analysis_text = f"生成分析失败：{e}"

    return jsonify({
        "chart_url": "/static/analysis.png",
        "analysis": analysis_text
    })


# 后台线程：实时更新库存
def inventory_updater():
    global inventory
    path = os.path.join(os.path.dirname(__file__), 'inventory.json')
    while True:
        try:
            data = json.load(open(path, 'r', encoding='utf-8'))
            for k in inventory: inventory[k] = int(data.get(k, inventory[k]))
        except Exception as e:
            print("[UPDATE_ERR]", e)
        time.sleep(1)


if __name__ == '__main__':
    threading.Thread(target=inventory_updater, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=True)