from flask import Flask, request, jsonify, render_template
from groq import Groq
import os, json, time
from collections import defaultdict

app = Flask(__name__)

user_profiles = defaultdict(lambda: {
    "messages": [], "risk_score": 0, "trajectory": "stable",
    "status": "safe", "warnings": 0, "banned": False
})

SYSTEM_PROMPT = """You are a content moderation AI. Analyze the given text and classify it.
Respond ONLY with a JSON object in this exact format, nothing else:
{
  "classification": "safe|spam|toxic|offensive",
  "confidence": {"safe": 0.0, "spam": 0.0, "toxic": 0.0, "offensive": 0.0},
  "toxicity_score": 0,
  "reason": "brief explanation"
}
toxicity_score is a number from 0 to 100.
All confidence values must add up to 1.0."""

def calculate_trajectory(messages):
    if len(messages) < 2: return "stable"
    recent = messages[-3:] if len(messages) >= 3 else messages
    scores = [m["toxicity_score"] for m in recent]
    if scores[-1] > scores[0] + 10: return "rising"
    elif scores[-1] < scores[0] - 10: return "falling"
    return "stable"

def calculate_risk(profile):
    messages = profile["messages"]
    if not messages: return 0
    avg_toxicity = sum(m["toxicity_score"] for m in messages) / len(messages)
    recent_toxicity = messages[-1]["toxicity_score"]
    trajectory_bonus = 20 if profile["trajectory"] == "rising" else 0
    warning_bonus = profile["warnings"] * 10
    return min(100, int((avg_toxicity * 0.4) + (recent_toxicity * 0.4) + trajectory_bonus + warning_bonus))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return jsonify({'error': 'API key not configured'}), 500
        client = Groq(api_key=api_key)
        data = request.get_json()
        text = data.get('text', '').strip()
        username = data.get('username', 'anonymous').strip()
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        if user_profiles[username]["banned"]:
            return jsonify({'blocked': True, 'reason': 'User is banned', 'username': username})
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this text: {text}"}
            ]
        )
        result = response.choices[0].message.content
        clean = result.strip().replace('```json', '').replace('```', '').strip()
        parsed = json.loads(clean)
        profile = user_profiles[username]
        profile["messages"].append({
            "text": text, "classification": parsed["classification"],
            "toxicity_score": parsed.get("toxicity_score", 0), "timestamp": time.time()
        })
        profile["trajectory"] = calculate_trajectory(profile["messages"])
        profile["risk_score"] = calculate_risk(profile)
        action = "allow"
        if profile["risk_score"] >= 80:
            profile["status"] = "banned"; profile["banned"] = True; action = "ban"
        elif profile["risk_score"] >= 50:
            profile["status"] = "warning"; profile["warnings"] += 1; action = "warn"
        elif profile["risk_score"] >= 25:
            profile["status"] = "watch"; action = "watch"
        else:
            profile["status"] = "safe"
        return jsonify({
            'blocked': action == "ban", 'action': action,
            'classification': parsed["classification"],
            'confidence': parsed["confidence"],
            'toxicity_score': parsed.get("toxicity_score", 0),
            'reason': parsed["reason"],
            'user': {
                'username': username, 'risk_score': profile["risk_score"],
                'trajectory': profile["trajectory"], 'status': profile["status"],
                'warnings': profile["warnings"], 'message_count': len(profile["messages"])
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/users', methods=['GET'])
def get_users():
    users = [{'username': u, 'risk_score': p["risk_score"], 'trajectory': p["trajectory"],
               'status': p["status"], 'warnings': p["warnings"],
               'message_count': len(p["messages"]), 'banned': p["banned"]}
              for u, p in user_profiles.items()]
    users.sort(key=lambda x: x["risk_score"], reverse=True)
    return jsonify(users)

@app.route('/reset/<username>', methods=['POST'])
def reset_user(username):
    if username in user_profiles: del user_profiles[username]
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
