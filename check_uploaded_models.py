from pprint import pprint
from huggingface_hub import ModelFilter, HfApi

api = HfApi()
filt = ModelFilter(author='m3')
models = api.list_models(filter=filt)
models_filtered = [i.modelId for i in models if 'text-classification' in i.tags]
models_ex = [i.modelId for i in models if 'text-classification' not in i.tags]

pprint(sorted([i for i in models_filtered if 'eda' in i]))
pprint(sorted([i for i in models_filtered if 'word-swapping-embedding' in i]))
pprint(sorted([i for i in models_filtered if 'word-swapping-synonym' in i]))
pprint(sorted([i for i in models_filtered if 'word-swapping-random' in i]))
pprint(sorted(models_ex))
