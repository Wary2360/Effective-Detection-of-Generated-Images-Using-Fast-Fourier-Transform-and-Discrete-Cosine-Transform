import timm
from pprint import pprint
# for model in timm.list_models():
#     if model.startswith('swin'):
#         print(model)

pprint(timm.list_models('*swin*'))