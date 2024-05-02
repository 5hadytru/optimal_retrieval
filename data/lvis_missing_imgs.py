import json, os
import requests

with open("data/LVIS/lvis_v1_train.json", "rb") as f:
    j = json.load(f)

count = 0
for t, i in enumerate(j["images"]):
    print(t)
    file_pth = 'data/LVIS/train2017/' + ''.join(['0' for _ in range(12 - len(str(i['id'])))]) + str(i['id']) + '.jpg'
    if not os.path.isfile(file_pth):
        dl_url = i["coco_url"]
        response = requests.get(dl_url)
        if response.status_code == 200:
            with open(file_pth, 'wb') as file:
                file.write(response.content)
        else:
            print(f"Failed to download image from {dl_url}")

"""
['train', 'val', 'test', 'imagenet_prompts', 'custom_prompts', 'best_7_imagenet_prompts', 'custom_cls_name_to_coco_cat_id']
{'label_names': ['raccoon'], 'width': 275, 'height': 183, 'boxes': tensor([[0.4036, 0.5273, 0.6545, 0.9344]]), 'labels': tensor([0.]), 'imagenet_prompts': ['a bad photo of the raccoon.'], 'custom_prompts': ['A photo of a raccoon with a striped tail']}
"""