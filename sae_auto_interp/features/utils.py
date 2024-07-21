from transformers import PreTrainedTokenizer
from torchtyping import TensorType

from .features import FeatureRecord

def display(
    record: FeatureRecord,
    tokenizer: PreTrainedTokenizer,
    threshold: float = 0.0,
) -> str:

    from IPython.core.display import display, HTML

    def _to_string(
        tokens: TensorType["seq"], 
        activations: TensorType["seq"]
    ) -> str:
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
            tokenizer.batch_decode(example.tokens), 
            example.activations
        )
        for example in record.examples
    ]

    display(HTML("<br><br>".join(strings)))