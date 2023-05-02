import requests
from PIL import Image

from ucall.client import Client

contents = [
    ('Steep brook in the forest', 'https://picsum.photos/id/28/4928/3264.jpg'),
    ('Alpine valley in sunlight', 'https://picsum.photos/id/29/4928/3264.jpg'),
    ('New York in black and white', 'https://picsum.photos/id/43/4928/3264.jpg')
]

client = Client()

for idx, content in enumerate(contents):
    image = Image.open(requests.get(content[1]).content)
    client.add(label=idx, description=content[0], photo=image)

query = 'pink panda'
response = client.search(query)
print(f'Closest results to "query" is:')
for idx, label in enumerate(list(response.numpy)):
    print(f'- {idx}. {contents[label]}')
