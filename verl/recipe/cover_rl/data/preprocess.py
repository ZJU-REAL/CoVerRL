import os

import datasets


def make_map_fn(split, source=None):
        def process_fn(example, idx):
            if source is None:
                data_source = example.pop("source")
            else:
                data_source = source
            question = example.pop("prompt")
            solution = example.pop("answer")
            

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": f"{data_source}-{idx}",
                    "answer": solution,
                    "question": question,
                },
            }
            return data

        return process_fn

if __name__ == '__main__':

    data_source = 'MATH-TTT'

    train_dataset = datasets.load_dataset("json", data_files=os.path.join(data_source, 'train.json'), split='train')
    test_dataset = datasets.load_dataset("json", data_files=os.path.join(data_source, 'test.json'), split='train')

    train_dataset = train_dataset.map(function=make_map_fn("train", data_source), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test", data_source), with_indices=True)

    train_dataset.to_parquet(os.path.join(data_source, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(data_source, 'test.parquet'))
