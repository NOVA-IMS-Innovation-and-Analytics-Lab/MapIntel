"""Configuration of MapInel system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import colorcet as cc

CONFIG: dict[str, dict[str, str | int]] = {
    'topics_mapping': {'cmap': cc.cm.gouldian},
    'qa_pipeline': {
        'prompt': (
            "Given the context please answer the question. If the answer is not contained in detail"
            "within the context below, then ignore the context and just answer with your own words and information"
            "\nContext: {join(documents)};\n Question: {query};\n Answer: "
        ),
        'max_length': 200,
    },
    'plot_map': {
        'wrap_width': 30,
        'wrap_length': 10,
    },
    'qa_answer': {
        'default_value': 'Ask me a question!',
    },
    'plot_prompt': {
        'placeholder_tokens': 'Provide a description',
        'placeholder_question': 'Ask a question',
    },
    'top_k': {
        'default_value': 10,
    },
}
