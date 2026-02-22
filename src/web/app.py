"""
光伏CAD Web界面
提供简单的用户交互界面
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import json
from src.inference.infer import PVCAIInference
import tempfile

# 创建Flask应用
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 全局推理实例
inference_instance = None

def get_inference_instance():
    """获取推理实例（懒加载）"""
    global inference_instance
    if inference_instance is None:
        try:
            inference_instance = PVCAIInference()
            print("✅ Inference instance created successfully")
        except Exception as e:
            print(f"❌ Failed to create inference instance: {e}")
            inference_instance = None
    return inference_instance

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_cad():
    """生成CAD设计"""
    try:
        # 获取上传的文件和文本
        if 'sketch' not in request.files:
            return jsonify({'error': 'No sketch image provided'}), 400
        
        sketch_file = request.files['sketch']
        text_description = request.form.get('description', 'Photovoltaic system design')
        
        if sketch_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # 保存上传的文件
        sketch_path = os.path.join(app.config['UPLOAD_FOLDER'], sketch_file.filename)
        sketch_file.save(sketch_path)
        
        # 获取推理实例
        inference = get_inference_instance()
        if inference is None:
            return jsonify({'error': 'Model initialization failed'}), 500
        
        # 生成CAD
        cad_result = inference.generate_cad(sketch_path, text_description)
        
        # 保存结果
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cad_result.json')
        with open(result_path, 'w') as f:
            json.dump(cad_result, f, indent=2)
        
        # 清理上传的文件
        os.remove(sketch_path)
        
        return jsonify({
            'success': True,
            'message': 'CAD generated successfully',
            'result_path': '/download/result'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/result')
def download_result():
    """下载CAD结果"""
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cad_result.json')
    if os.path.exists(result_path):
        return send_file(result_path, as_attachment=True, download_name='pv_cad_result.json')
    else:
        return jsonify({'error': 'Result file not found'}), 404

@app.route('/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': inference_instance is not None
    })

if __name__ == '__main__':
    # 创建模板目录
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    # 创建简单的HTML模板
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>PV CAD AI Designer</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { border: 1px solid #ccc; padding: 20px; border-radius: 10px; }
            input[type="file"], input[type="text"], button { margin: 10px 0; padding: 10px; width: 100%; }
            button { background: #4CAF50; color: white; border: none; cursor: pointer; }
            button:hover { background: #45a049; }
            .result { margin-top: 20px; padding: 10px; background: #f9f9f9; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>光伏CAD智能设计系统</h1>
            <form id="cadForm" enctype="multipart/form-data">
                <input type="file" id="sketch" name="sketch" accept="image/*" required>
                <input type="text" id="description" name="description" placeholder="输入文字描述，例如：住宅屋顶光伏系统设计" required>
                <button type="submit">生成CAD设计</button>
            </form>
            <div id="result" class="result" style="display:none;"></div>
        </div>
        <script>
            document.getElementById('cadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const resultDiv = document.getElementById('result');
                
                try {
                    const response = await fetch('/generate', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        resultDiv.innerHTML = `<p>✅ ${result.message}</p><a href="${result.result_path}" target="_blank">下载CAD结果</a>`;
                        resultDiv.style.display = 'block';
                    } else {
                        resultDiv.innerHTML = `<p>❌ 错误: ${result.error}</p>`;
                        resultDiv.style.display = 'block';
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<p>❌ 网络错误: ${error.message}</p>`;
                    resultDiv.style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    '''
    
    with open(os.path.join(template_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ Web application ready!")
    print("访问 http://localhost:5000 开始使用")
    app.run(host='0.0.0.0', port=5000, debug=True)