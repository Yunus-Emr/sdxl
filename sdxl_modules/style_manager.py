import json
def load_styles(json_path="SDXL_Test/styles.json"):
    with open(json_path, "r") as f:
        styles = json.load(f)
    return styles

def get_style_prompt(style, prompt_text):
    return style["prompt"].format(prompt=prompt_text), style["negative_prompt"]