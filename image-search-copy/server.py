import numpy as np
from PIL import Image

import usearch
import uform
from ucall.rich_posix import Server

server = Server()
model = uform.get_model('unum-cloud/uform-vl-multilingual')
index = usearch.Index(dim=256)


@server
def add(label: int, photo: Image.Image, description: str):
    image = model.preprocess_image(photo)
    tokens = model.preprocess_text(description)
    vector = model.encode_multimodal(image=image, text=tokens).detach().numpy()
    labels = np.array([label], dtype=np.longlong)
    index.add(labels, vector, copy=True)


@server
def search(query: str) -> np.ndarray:
    tokens = model.preprocess_text(query)
    vector = model.encode_text(tokens).detach().numpy()
    neighbors = index.search(vector, 5)
    return neighbors[0][:neighbors[2][0]]


server.run()
