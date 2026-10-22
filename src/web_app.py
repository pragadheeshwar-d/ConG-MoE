import os
import json
import time
import pickle
from http.server import HTTPServer, BaseHTTPRequestHandler
from predict_interactive import get_or_train_model, predict_comment, STATE_EMOJIS, TARGET_CLASSES

PORT = 5050
MODEL = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tanglish Sentiment Predictor (ConG-MoE v2)</title>
    <style>
        :root {
            --bg: #0f172a;
            --card: #1e293b;
            --border: #334155;
            --text: #f8fafc;
            --muted: #94a3b8;
            --primary: #38bdf8;
            --accent: #818cf8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 40px 20px; display: flex; justify-content: center; }
        .container { max-width: 720px; width: 100%; }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 32px; box-shadow: 0 10px 25px rgba(0,0,0,0.3); }
        h1 { font-size: 26px; font-weight: 700; margin-bottom: 8px; color: var(--primary); }
        p.subtitle { color: var(--muted); font-size: 14px; margin-bottom: 24px; }
        .input-group { margin-bottom: 20px; }
        label { display: block; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); margin-bottom: 8px; }
        textarea {
            width: 100%;
            height: 110px;
            background: #0b1120;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px;
            color: var(--text);
            font-size: 16px;
            resize: vertical;
            outline: none;
            transition: border-color 0.2s;
        }
        textarea:focus { border-color: var(--primary); }
        .btn-row { display: flex; gap: 12px; margin-bottom: 24px; }
        button {
            flex: 1;
            background: linear-gradient(135deg, #0284c7, #6366f1);
            color: white;
            border: none;
            padding: 14px 24px;
            font-size: 15px;
            font-weight: 600;
            border-radius: 10px;
            cursor: pointer;
            transition: transform 0.1s, opacity 0.2s;
        }
        button:hover { opacity: 0.95; }
        button:active { transform: scale(0.98); }
        .sample-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px; }
        .chip {
            background: #334155;
            color: #cbd5e1;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            cursor: pointer;
            border: 1px solid transparent;
            transition: all 0.2s;
        }
        .chip:hover { background: #475569; border-color: var(--primary); color: white; }
        #result-box { display: none; margin-top: 24px; border-top: 1px solid var(--border); padding-top: 24px; }
        .badge {
            display: inline-block;
            font-size: 18px;
            font-weight: 700;
            padding: 10px 18px;
            border-radius: 10px;
            margin-bottom: 16px;
        }
        .badge-Positive { background: #064e3b; color: #34d399; border: 1px solid #059669; }
        .badge-Negative { background: #4c0519; color: #fb7185; border: 1px solid #e11d48; }
        .badge-Mixed_feelings { background: #451a03; color: #fbbf24; border: 1px solid #d97706; }
        .badge-Unknown_state { background: #1e1b4b; color: #a5b4fc; border: 1px solid #4f46e5; }
        .meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px; font-size: 13px; }
        .meta-item { background: #0b1120; padding: 12px; border-radius: 8px; border: 1px solid #1e293b; }
        .meta-label { color: var(--muted); font-size: 11px; text-transform: uppercase; margin-bottom: 4px; }
        .meta-val { font-weight: 600; color: #e2e8f0; }
        .bar-container { margin-bottom: 12px; }
        .bar-header { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; }
        .bar-bg { background: #0b1120; height: 10px; border-radius: 5px; overflow: hidden; }
        .bar-fill { height: 100%; border-radius: 5px; background: var(--primary); transition: width 0.4s ease; }
        .alert-box {
            background: #1e1b4b;
            border-left: 4px solid var(--accent);
            padding: 12px 16px;
            border-radius: 0 8px 8px 0;
            font-size: 13px;
            color: #c7d2fe;
            margin-top: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>Tanglish Sentiment Predictor</h1>
            <p class="subtitle">ConG-MoE v2: 3-Head Multi-Expert Classifier for Code-Mixed Social Comments</p>

            <div class="input-group">
                <label>Enter Tanglish Comment / Review:</label>
                <textarea id="comment" placeholder="e.g. Padam sema mass aana second half romba bore da 🤮"></textarea>
            </div>

            <div class="sample-chips">
                <span class="chip" onclick="setSample('Thalaivar entry vera level mass 🔥 padam blockbuster hit confirmed!')">🌟 Pure Praise</span>
                <span class="chip" onclick="setSample('Padam romba mokka da timing waste don\'t go theatre 🤮')">💔 Pure Criticism</span>
                <span class="chip" onclick="setSample('acting sema aana screenplay worst')">⚖️ Twist ('aana')</span>
                <span class="chip" onclick="setSample('first half nalla irukku aana climax mokka 🤮')">⚖️ Twist + Emoji</span>
                <span class="chip" onclick="setSample('GOAT trailer vera level bro... eppo ott release date???')">❓ Release Date Inquiry</span>
            </div>

            <div class="btn-row">
                <button onclick="runPredict()">⚡ Predict Sentiment State</button>
            </div>

            <div id="result-box">
                <div id="state-badge" class="badge"></div>
                <div class="meta-grid">
                    <div class="meta-item">
                        <div class="meta-label">Inference Latency</div>
                        <div class="meta-val" id="meta-latency">-</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">Base Confidence</div>
                        <div class="meta-val" id="meta-conf">-</div>
                    </div>
                </div>

                <div id="override-box" class="alert-box" style="display:none;"></div>

                <label style="margin-top: 20px;">Class Probabilities:</label>
                <div id="bars"></div>
            </div>
        </div>
    </div>

    <script>
        function setSample(text) {
            document.getElementById('comment').value = text;
            runPredict();
        }

        async function runPredict() {
            const comment = document.getElementById('comment').value.trim();
            if (!comment) return;

            const resBox = document.getElementById('result-box');
            resBox.style.display = 'block';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ comment })
                });
                const data = await response.json();

                const badge = document.getElementById('state-badge');
                badge.className = 'badge badge-' + data.final_state;
                badge.innerText = data.state_label;

                document.getElementById('meta-latency').innerText = data.latency_ms.toFixed(2) + ' ms (CPU)';
                document.getElementById('meta-conf').innerText = (data.confidence * 100).toFixed(1) + '%';

                const overBox = document.getElementById('override-box');
                if (data.override_reason) {
                    overBox.style.display = 'block';
                    overBox.innerHTML = '<strong>Specialist Intervened:</strong> ' + data.override_reason;
                } else {
                    overBox.style.display = 'none';
                }

                const barsContainer = document.getElementById('bars');
                barsContainer.innerHTML = '';
                const colors = {
                    'Positive': '#10b981',
                    'Negative': '#f43f5e',
                    'Mixed_feelings': '#f59e0b',
                    'Unknown_state': '#6366f1'
                };

                for (const [cls, prob] of Object.entries(data.prob_dict)) {
                    const pct = (prob * 100).toFixed(1);
                    const color = colors[cls] || '#38bdf8';
                    barsContainer.innerHTML += `
                        <div class="bar-container">
                            <div class="bar-header">
                                <span>${cls}</span>
                                <strong>${pct}%</strong>
                            </div>
                            <div class="bar-bg">
                                <div class="bar-fill" style="width: ${pct}%; background: ${color};"></div>
                            </div>
                        </div>
                    `;
                }
            } catch (err) {
                alert('Error connecting to backend: ' + err);
            }
        }
    </script>
</body>
</html>
"""

class TanglishServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path == "/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)
            comment = data.get("comment", "")

            res = predict_comment(comment, MODEL)
            res["state_label"] = STATE_EMOJIS.get(res["final_state"], res["final_state"])

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

def run():
    global MODEL
    MODEL = get_or_train_model()
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, TanglishServer)
    print(f"\n========================================================")
    print(f"  Tanglish Predictor Web App is live at: http://localhost:{PORT}")
    print(f"  Open this URL in your web browser!")
    print(f"========================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")

if __name__ == "__main__":
    run()
