
"""
Evaluate model correctness
"""
import numpy as np
import tqdm
import config
import os
import pickle
import evaluate
import transformers
from sentence_transformers import CrossEncoder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run-name', type=str, default='run_1')
args = parser.parse_args()
run_name = args.run_name


class CrossEncoderSimilarity:
    def __init__(
            self,
            model_id="cross-encoder/stsb-distilroberta-base",
            device="cuda",
            weight=1
        ):
        self.model = CrossEncoder(model_id, device=device)
        self.weight = weight

    def __call__(self, sources=None, summaries=None):
        scores = self.model.predict([[src, sum] for src, sum in zip(sources, summaries)])
        return scores.tolist()

def eval_correctness():
    # criterion = evaluate.load("squad")
    criterion = CrossEncoderSimilarity()
    with open(os.path.join(config.output_dir, run_name, 'generations.pkl'), 'rb') as f:
        generations = pickle.load(f)
    scores = []
    for gen in tqdm.tqdm(generations):
        prediction = gen['most_likely_generation'].lstrip()
        if prediction[-1] == '.':
            prediction = prediction[:-1]
        answers = gen['answer'] + gen['additional_answers'] if gen['additional_answers'] is not None else gen['answer']
        max_score = 0.0
        for answer in answers:
            if answer[-1] == '.':
                answer = answer[:-1]
            if isinstance(answer, str) and isinstance(prediction, str) and len(answer) > 0 and len(prediction) > 0:
                results = criterion(sources=answer, summaries=prediction)
            else:
                results = [0]
            max_score = max(results[0], max_score)
            # print(prediction, answer, results)
        scores.append(max_score)

    for i in np.arange(0.1, 1.1, 0.1):
        print(i, (np.asarray(scores) > i).sum() / len(scores))

    with open(os.path.join(config.output_dir, run_name, 'sentsim_for_correctness.pkl'), 'wb') as f:
        pickle.dump(scores, f)



if __name__ == '__main__':
    eval_correctness()
