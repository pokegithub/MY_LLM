import types


def fake_transformers_module(output_text: str):
    return fake_transformers_sequence([output_text])


def fake_transformers_sequence(output_texts):
    outputs = list(output_texts)
    state = {"index": 0}

    class FakeTokenizer:
        model_max_length = 128
        chat_template = None

        def __call__(self, prompt, return_tensors=None, **kwargs):
            return {"input_ids": [[1, 2, 3]]}

        def decode(self, generated, skip_special_tokens=True):
            index = min(state["index"], len(outputs) - 1)
            value = outputs[index]
            state["index"] += 1
            return value

    class FakeModel:
        def eval(self):
            return self

        def generate(self, **kwargs):
            return [[1, 2, 3, 4]]

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeTokenizer()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeModel()

    return types.SimpleNamespace(
        __version__="fake",
        AutoTokenizer=AutoTokenizer,
        AutoModelForCausalLM=AutoModelForCausalLM,
    )
