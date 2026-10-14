from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase
import torch


class RanksftDataCollator(DataCollatorForSeq2Seq):
    def __init__(self, *args, **kwargs):
        self.mask_rel_token = kwargs.pop("mask_rel_token", False)
        super().__init__(*args, **kwargs)
        self.judgment_toks = self.tokenizer.convert_tokens_to_ids(["<IRRELEVANT>", "<RELEVANT>"])

    def __call__(self, features, return_tensors=None):
        classes = []
        ans_score = []
        if "grouped_inputs" in features[0].keys():
            new_features = []
            for feature in features:
                new_features += feature["grouped_inputs"]
            features = new_features
        for feature in features:
            classes.append(feature.pop("classes"))
            ans_score.append(feature.pop("ans_score"))
        features = super().__call__(features, return_tensors=return_tensors)
        if self.mask_rel_token:
            for label_idx, label in enumerate(features['input_ids']):
                for idx, tok in enumerate(label):
                    if tok in self.judgment_toks:
                        features['labels'][label_idx][idx] = self.label_pad_token_id
                        break
        features['classes'] = torch.tensor(classes, dtype=torch.float16)
        if ans_score[0] == [] or ans_score[0] is None:
            features['ans_score'] = None
        else:
            features['ans_score'] = torch.tensor(ans_score, dtype=torch.float16)
        return features
