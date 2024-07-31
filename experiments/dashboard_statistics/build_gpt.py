# %%

import html
import os
import re
from collections import defaultdict

import orjson

explanation_dir = "results/gpt2_top/gpt2_explanations"
recall_dir = "results/gpt2_top/gpt2_recall"
fuzz_dir = "results/gpt2_top/gpt2_fuzz"

data = defaultdict(dict)


def create_html_content(feature_name, explanation, recall_scores, fuzz_scores):
    prompt = _highlight(
        explanation.get("generation_prompt", "No prompt available.")[1:], 1.0
    )
    prompt = prompt.replace("\n", "<br>")
    response = explanation.get("response", "No response available.").replace(
        "\n", "<br>"
    )

    feature_name = edit(feature_name)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{feature_name}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen-Sans, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #FBFBFB;
            }}
            h1, h2 {{
                color: #2c3e50;
                margin-top: 0px;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            .subtitle {{
                font-size: 10px;
                color: #666;
                margin-top: 5px;
                margin-bottom: 0px;
            }}
            .card {{
                background-color: #ffffff;
                border-radius: 8px;
                border: solid #E4E4E7;
                border-width: .25px;
                padding: 20px;
                margin-bottom: 20px;
            }}
            .explanation {{
                background-color: #e8f6f3;
                border-left: 4px solid #1abc9c;
                padding: 3px 15px 3px 15px;
                margin-top: 20px;
            }}
            .score-item {{
                margin-bottom: 5px;
                display: flex;
                align-items: center;
            }}
            .highlighted {{
                background-color: #ffff99;
            }}
            .circle {{
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                margin-right: 5px;
                border: 0.5px solid #84848C;
            }}
            .true {{
                background-color: #27ae60;
            }}
            .false {{
                background-color: #c0392b;
            }}
            .none {{
                background-color: #ffffff;
            }}
            .key {{
                font-size: 10px;
                color: #84848C;
            }}
        </style>
    </head>
    <body>
        <h1>Feature Analysis: {feature_name}</h1>

        <div class="card">
            <h2>Model Generation</h2>
            <h3>Prompt:</h3>
            {prompt}
            <h3>Response:</h3>
            {response}
        </div>

        <div class="card">
            <h2>Explanation</h2>
            <div class="explanation">
                <p>{html.escape(explanation.get('explanation', 'No explanation available.'))}</p>
            </div>
        </div>

        <div class="card">
            <h2>
                Scores
                <p class="subtitle">
                    ● Recall | ● Fuzz | # Distance
                    <br>
                    Green indicates a correct prediction, red incorrect.
                    For recall, -1 indicates a non-activating example.
                    For fuzzing, -1 indicates an incorrectly marked example.
                </p>
            </h2>
            <div id="scores">
                {generate_score_html(recall_scores, fuzz_scores)}
            </div>
        </div>
    </body>
    </html>
    """
    return html_content, feature_name


def _highlight(text, opacity: float = 1.0):
    return text.replace(
        "<<", f'<span style="background-color:rgba(57,212,155, {opacity})">'
    ).replace(">>", "</span>")


def generate_score_html(recall_scores, fuzz_scores):
    results = {}
    fuzz_wrong = []
    recall_wrong = []

    for score in fuzz_scores:
        s = {
            "ground_truth": score["ground_truth"],
            "fuzz": score["prediction"],
            "distance": score["distance"],
            "text": score["text"],
            "recall": None,
        }

        if s["ground_truth"] is False:
            fuzz_wrong.append(s)
        else:
            results[score["id"]] = s

    for score in recall_scores:
        if score["ground_truth"] is False:
            s = {
                "ground_truth": score["ground_truth"],
                "fuzz": None,
                "distance": score["distance"],
                "text": score["text"],
                "recall": score["prediction"],
            }

            recall_wrong.append(s)

            continue

        s = results[score["id"]]
        s["recall"] = score["prediction"]

    sorted_scores = sorted(list(results.values()), key=lambda x: x["distance"])
    sorted_scores += recall_wrong[:10]
    sorted_scores += fuzz_wrong[:10]

    score_html = ""
    correct = True
    for score in sorted_scores:
        if score["distance"] == -1 and correct:
            correct = False
            score_html += "<hr>"

        text = score["text"]

        opacity = (
            0.2 + 0.8 * (abs(score["distance"] - 11) / 10)
            if score["distance"] != -1
            else 0.2
        )
        text = _highlight(text, opacity)

        match score["recall"]:
            case None:
                recall = "none"
            case False:
                recall = "false"
            case True:
                recall = "true"

        recall_circle = f'<span class="circle {recall}"></span>'

        match score["fuzz"]:
            case None:
                fuzz = "none"
            case False:
                fuzz = "false"
            case True:
                fuzz = "true"

        fuzz_circle = f'<span class="circle {fuzz}"></span>'

        score_html += f"""
        <div class="score-item">
            <div style="min-width: 60px;">
                {recall_circle}{fuzz_circle}
                <small>{score['distance']}</small>
            </div>
            <span>{text}</span>
        </div>
        """
    return score_html


def transform_string(input_string):
    pattern = re.compile(r"\.transformer\.h\.(\d+)_feature(\w+)")
    match = pattern.search(input_string)

    if match:
        layer = match.group(1)
        feature = match.group(2)
        return f"resid_{layer}-{feature}"
    else:
        raise ValueError("Input string does not match the expected format")


def edit(path):
    return transform_string(path)


def build_html_page(data):
    os.makedirs("gpt_top_html", exist_ok=True)
    for feature, feature_data in data.items():
        html_content, feature_name = create_html_content(
            feature,
            feature_data["explanation"],
            feature_data["recall"],
            feature_data["fuzz"],
        )
        with open(f"gpt_top_html/{feature_name}.html", "w", encoding="utf-8") as f:
            f.write(html_content)
    print("HTML files have been created in the 'gpt_top_html' directory.")


# Process all files
for file in os.listdir(explanation_dir):
    feature_name = file.replace(".txt", "")

    try:
        with open(f"{explanation_dir}/{file}", "rb") as f:
            explanation = orjson.loads(f.read())

        with open(f"{recall_dir}/{file}", "rb") as f:
            recall = orjson.loads(f.read())

        with open(f"{fuzz_dir}/{file}", "rb") as f:
            fuzz = orjson.loads(f.read())

    except Exception:
        print(file)
        continue

    data[feature_name] = {"explanation": explanation, "recall": recall, "fuzz": fuzz}

# Generate HTML files
build_html_page(data)
