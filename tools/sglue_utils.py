import re

# These weights are based on the number of examples in each dataset.
SUPER_GLUE_WEIGHT_MAPPING = {
    "dpr_v001_simple": 1_322.,
    "super_glue_wsc_v102_simple_train": 259.,
    "super_glue_wsc_v102_simple_eval": 0.,
    "super_glue_boolq_v102": 9_427.,
    "super_glue_cb_v102": 250.,
    "super_glue_copa_v102": 400.,
    "super_glue_multirc_v102": 27_243.,
    "super_glue_record_v102": 138_854.,
    "super_glue_rte_v102": 2_490.,
    "super_glue_wic_v102": 5_428.,
    "super_glue_axb_v102": 0.,
    "super_glue_axg_v102": 0.,
}


def get_super_glue_text_preprocessor(task_name):
    if task_name == "BoolQ":
        return preprocess_boolq
    elif task_name == "CB":
        return preprocess_cb
    elif task_name == "COPA":
        return preprocess_copa
    elif task_name == "MultiRC":
        return preprocess_multirc
    elif task_name == "ReCoRD":
        return preprocess_record
    elif task_name == "RTE":
        return preprocess_rte
    elif task_name == "WiC":
        return preprocess_wic
    elif task_name == "WSC":
        return preprocess_wsc


def expose_features(x, features):
    feature_string = []
    for key in features:
        feature_string.append("{}: {}".format(key, x[key]))

    return " ".join(feature_string)


def preprocess_boolq(x):

    joined = " ".join([
        "boolq",
        expose_features(x, ["hypothesis", "premise"])
    ])
    return {
        "text": joined,
        "target": x["label"],
    }


def preprocess_cb(x):

    joined = " ".join([
        "cb",
        expose_features(x, ["hypothesis", "premise"])
    ])
    return {
        "text": joined,
        "target": x["label"],
    }


def preprocess_copa(x):

    joined = " ".join([
        "copa",
        expose_features(x, ["choice1", "choice2", "premise", "question"])
    ])
    return {
        "text": joined,
        "target": ["False", "True"][x["label"]],
    }


def preprocess_multirc(x):

    processed_line = []
    text = x["passage"]["text"]
    # Remove HTML markup.
    text = tf.strings.regex_replace('<br>', ' ', text)
    text = tf.strings.regex_replace('<(/)?b>', '', text)

    for qa_pair in x["passage"]["questions"]:
        question = qa_pair["question"]
        for answer in qa_pair["answers"]:
            processed_line.append({
                "text": "multirc question: {} answer: {} paragraph: {}".format(
                    question, answer["text"], text
                    ),
                "target": ["False", "True"][answer["label"]],
            })

    return processed_line


def preprocess_record(x):

    processed_line = []
    passage = x["passage"]["text"]
    passage = re.sub(r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', 
        passage)
    passage = re.sub(r'\n@highlight\n', '. ', 
        passage)

    entities = []
    for idx in line["passage"]['entities']:
        entities.append(passage[idx["start"]:idx["end"]+1])
    entities = " ".join(entities)

    for qas in x["qas"]:
        for answer in qas["answers"]:
            processed_line.append({
                "text": "record query: {} entities: {} passage: {}".format(
                    qas["query"], entities, passage
                    ),
                "target": answer["text"],
            })

    return processed_line


def preprocess_rte(x):

    joined = " ".join([
        "rte",
        expose_features(x, ["sentence1", "sentence2"])
    ])
    return {
        "text": joined,
        "target": x["label"],
    }


def preprocess_wic(x):

    joined = " ".join([
        "wsc",
        expose_features(x, ["sentence1", "sentence2", "word"])
    ])
    return {
        "text": joined,
        "target": str(x["label"]),
    }


def preprocess_wsc(x):

    def _mark_span(text, span_str, span_idx, mark):

        pattern_tmpl = r'^((?:\S+\s){N})(W)'
        pattern = re.sub(pattern_tmpl, 'N', str(span_idx))
        pattern = re.sub(pattern, 'W', span_str)
        return re.sub(pattern, r'{}\g<0>{}'.format(mark, mark), text)

    text = x["text"]
    text = _mark_span(text, x["target"]["span1_text"], x["target"]["span1_index"], "*")

    # Compensate for 2 added "words" added in previous step.
    span2_index = x["target"]["span2_index"] \
        + 2 * int(x["target"]["span1_index"] < x["target"]["span2_index"])

    text = _mark_span(text, x["target"]["span2_text"], span2_index, '#')

    joined = " ".join(["wsc", "text:", text])

    if "label" in x:
        if x["label"] == -1:
            label_name = "<unk>"
        else:
            label_name = x["label"]
    else:
        label_name = "<unk>"

    return {"text": joined, "target": label_name}
