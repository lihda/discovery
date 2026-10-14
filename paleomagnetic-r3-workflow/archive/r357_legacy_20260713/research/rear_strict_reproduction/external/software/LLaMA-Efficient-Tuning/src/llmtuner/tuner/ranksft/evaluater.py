import json
import torch
from transformers import GenerationConfig
from tqdm import tqdm

class Evaluator:
    generation_config = GenerationConfig.from_dict({"bos_token_id": 1,
                                                    "eos_token_id": 2,
                                                    "pad_token_id": 0,
                                                    "num_beams": 1,
                                                    "num_beam_groups": 1,
                                                    "temperature": 0,
                                                    "num_return_sequences": 1,
                                                    "do_sample": False,})
    
    data = json.loads(open("/mnt/wangyuhao/usere/eval/nq-test-1.json"))
    def evaluate(self, model, tokenizer, outfile, generation_config=None, data=None, device=None):
        if device is None:
            device = self.model.device
        if generation_config is None:
            generation_config = self.generation_config
        if data is None:
            data = self.data
        def tokenize(prompt):
            inputs = tokenizer(prompt, padding=True, truncation=True, add_special_tokens=False)
            return inputs

        def batch_generate(prompt):
            input_ids = tokenize(prompt).input_ids
            output_ids = model.generate(
                input_ids=torch.as_tensor(input_ids).to(device),
                generation_config=generation_config,
            )
            output_ids = [output_id[len(input_id):] for output_id, input_id in zip(output_ids, input_ids)]
            response = [tokenizer.decode(output_id, skip_special_tokens=True).strip() for output_id in output_ids]
            return response

        def batch_score(prompt):
            prompt = tokenize(prompt)
            inputs = {
                "input_ids": torch.as_tensor(prompt.input_ids).to(device),
                "attention_mask": torch.as_tensor(prompt.attention_mask).to(device),
                "return_rel_score": True,
            }
            with torch.no_grad():
                res = model(**inputs).rel_scores.tolist()
            return res

        def get_scores(query, ctxs, **kwargs):
            template = "<s>{document}<BEGIN_QUERY>{query}<GENERATE_SCORE>"
            batch_size = 10

            prompt = [template.format(document=doc, query=query) for doc in ctxs]
            scores = []
            for batch in range(0, len(ctxs), batch_size):
                scores += batch_score(prompt[batch: batch + batch_size])
            
            pos_prompt = [p + "<RELEVANT>" for p in prompt]
            pos_res = []
            for batch in range(0, len(ctxs), batch_size):
                pos_res += batch_generate(pos_prompt[batch: batch + batch_size])

            neg_prompt = [p + "<IRRELEVANT>" for p in prompt]
            neg_res = []
            for batch in range(0, len(ctxs), batch_size):
                neg_res += batch_generate(neg_prompt[batch: batch + batch_size])

            res = []
            for gold_pred, hard_pred, s in zip(pos_res, neg_res, scores):
                res.append({"preds": [gold_pred, hard_pred],
                            "scores": [[s, 0], [s, 0]]})
            return res
        
        def process(dic):
            ranksft_res = get_scores(dic['question'], dic['dense_ctxs'][:10])
            return {
                'question': dic['question'],
                'reference': dic['reference'],
                'dense_ctxs': dic['dense_ctxs'][:10],
                'ranksft_res': ranksft_res,
            }
        
        res = []
        for dic in tqdm(data[:100]):
            res.append(process(dic))

        with open(outfile, "w") as f:
            json.dump(res, f, ensure_ascii=False)