def display(
    examples=None,
    threshold=0.0,
) -> str:
    assert hasattr(examples[0], "str_toks"), \
        "Examples must have be detokenized to display."

    from IPython.core.display import display, HTML

    def _to_string(tokens, activations):
        result = []
        i = 0

        max_act = max(activations)
        _threshold = max_act * threshold

        while i < len(tokens):
            if activations[i] > _threshold:
                result.append("<mark>")
                while i < len(tokens) and activations[i] > _threshold:
                    result.append(tokens[i])
                    i += 1
                result.append("</mark>")
            else:
                result.append(tokens[i])
                i += 1
        return "".join(result)
    
    strings = [
        _to_string(
            example.str_toks, 
            example.activations
        ) 
        for example in examples
    ]

    display(HTML("<br><br>".join(strings)))