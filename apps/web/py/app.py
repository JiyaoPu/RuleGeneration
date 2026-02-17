from flask import Flask, request, jsonify, Response
import subprocess
import os
import json
import threading
import queue
import pandas as pd
from flask_cors import CORS
from flask import send_file

app = Flask(__name__)
CORS(app)

# 全局队列，用于存储 Experiment.py 推送的实时更新通知
update_queue = queue.Queue()

def run_experiment_process(args, env):
    """
    启动 Experiment.py 进程，并实时读取其 stdout。
    如果输出以 "UPDATE_EXCEL:" 开头，则说明 Excel 文件已更新，
    将此通知放入队列中；如果输出以 "UPDATE:" 开头，则解析并放入队列。
    """
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=env, text=True, encoding='utf-8', errors='replace')
    for line in process.stdout:
        line = line.strip()
        if line.startswith("UPDATE_EXCEL:"):
            try:
                # 收到Excel更新通知
                update_queue.put({"excel_updated": True})
                print("==++ Excel Updated  ++==")
            except Exception as e:
                print("解析 Excel 更新通知失败：", e)
        elif line.startswith("UPDATE:"):
            try:
                update_data = json.loads(line[len("UPDATE:"):])
                update_queue.put(update_data)
            except Exception as e:
                print("解析更新数据失败：", e)
        else:
            print("STDOUT:", line)
    process.stdout.close()
    process.wait()
    # 当进程结束时，放入结束标记
    update_queue.put({"done": True})

@app.route('/run_program', methods=['POST'])
def run_program():
    settings = request.json
    print("收到 settings:", settings)

    # 构造命令行参数
    args = [
        'python', 
        r'C:/Users/hilab/OneDrive/Desktop/Rule_Generation/PythonCode/variableMaze_Jiyao/Experiment.py',
        '--random_count', str(settings.get('input_0', 0)),
        '--cheater_count', str(settings.get('input_1', 0)),
        '--cooperator_count', str(settings.get('input_2', 0)),
        '--copycat_count', str(settings.get('input_3', 0)),
        '--grudger_count', str(settings.get('input_4', 0)),
        '--detective_count', str(settings.get('input_5', 0)),
        '--ai_count', str(settings.get('input_6', 0)),
        '--human_count', str(settings.get('input_7', 0)),
        '--trade_rules', 
            str(settings.get('input_8', 0)),
            str(settings.get('input_9', 0)),
            str(settings.get('input_10', 0)),
            str(settings.get('input_11', 0)),
            str(settings.get('input_12', 0)),
            str(settings.get('input_13', 0)),
        '--round_number', str(settings.get('roundNumberInput', 1)),
        '--reproduction_number', str(settings.get('reproductionNumberInput', 1)),
        '--mistake_possibility', str(settings.get('mistakePossibilityInput', 0)),
        '--fixed_rule', str(settings.get('hiddenFixedRule', True)),
        '--humanPlayer', str(settings.get('hiddenHumanPlayer', False)),
        '--ai_type', settings.get('hiddenAIType', 'Q'),
        '--cooperationRate', str(settings.get('cooperationRateInput', 0)),
        '--individualIncome', str(settings.get('individualIncomeInput', 0)),
        '--giniCoefficient', str(settings.get('giniCoefficientInput', 0)),
        '--batch_size', str(settings.get('de_input_batch_size', 1)),
        '--lr', str(settings.get('de_input_lr', 0.01)),
        '--b1', str(settings.get('de_input_b1', 0.5)),
        '--b2', str(settings.get('de_input_b2', 0.999)),
        '--RuleDimension', str(settings.get('de_input_RuleDimension', 17)),
        '--DE_train_episode', str(settings.get('de_input_DE_train_episode', 1)),
        '--DE_test_episode', str(settings.get('de_input_DE_test_episode', 1)),
        '--layersNum', str(settings.get('de_input_layersNum', 1)),
        '--evaluationSize', str(settings.get('de_input_evaluationSize', 1)),
        '--agent_train_epoch', str(settings.get('input_agent_train_epoch', 10000)),
        '--gamma', str(settings.get('input_gamma', 0.99)),
        '--epsilon', str(settings.get('input_epsilon', 1.0)),
        '--epsilon_decay', str(settings.get('input_epsilon_decay', 0.999)),
        '--epsilon_min', str(settings.get('input_epsilon_min', 0.1)),
        '--memory_size', str(settings.get('input_memory_size', 10000)),
        '--target_update', str(settings.get('input_target_update', 10)),
        '--state_size', str(settings.get('input_state_size', 20))
    ]
    print("执行命令：", " ".join(args))

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    # 启动后台线程运行 Experiment.py
    thread = threading.Thread(target=run_experiment_process, args=(args, env), daemon=True)
    thread.start()

    return jsonify({'message': 'Experiment started'})


@app.route("/q_table_heatmap.png")
def get_q_table_heatmap():
    return send_file("C:/Users/hilab/OneDrive/Desktop/Rule_Generation/WebDash/data/q_table_heatmap.png", mimetype="image/png")

@app.route('/get_test_results', methods=['GET'])
def get_test_results():
    """
    从本地 Excel 文件读取测试结果数据，并返回 JSON 数组。
    Excel 格式要求为：
      epoch, gini_coefficient, cooperation_rate, individual_income
    其中 epoch 为数字，其它3个字段为 JSON 字符串形式的9维列表。
    """
    excel_path = "C:/Users/hilab/OneDrive/Desktop/Rule_Generation/WebDash/data/test_results.xlsx"
    try:
        df = pd.read_excel(excel_path)
        # 将 DataFrame 转换为字典列表
        data = df.to_dict(orient='records')
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stream')
def stream():
    """ SSE接口，用于前端与后端保持连接，接收 Excel 更新或其他通知 """
    def event_stream():
        while True:
            try:
                data = update_queue.get(timeout=30)
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("done"):
                    break
            except queue.Empty:
                yield "data: {}\n\n"
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/get_excel_data', methods=['GET'])
def get_excel_data():
    """ 提供读取本地 dataupdates.xlsx 的接口，返回 JSON """
    try:
        df = pd.read_excel("C:/Users/hilab/OneDrive/Desktop/Rule_Generation/WebDash/data/dataupdates.xlsx")
        data = df.to_dict(orient='records')
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
