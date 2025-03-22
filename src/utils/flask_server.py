from flask import Flask, send_from_directory, render_template_string
import os

app = Flask(__name__)

MODELS_DIR = "/app/models"

@app.route("/")
def index():
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]
    html = """
    <h2>📦 Available Models for Download:</h2>
    <ul>
    {% for file in files %}
      <li><a href="/download/{{ file }}">{{ file }}</a></li>
    {% endfor %}
    </ul>
    """
    return render_template_string(html, files=files)

@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(MODELS_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
