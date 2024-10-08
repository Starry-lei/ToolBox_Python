from gensim.models import Word2Vec
import json
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# conda activate Cap3D    python word2Vector.py

part_names = [
    'backrest', 'leg', 'pillow', 'seat_frame', 'seat',
    'armrest_connector', 'seat_connector', 'foot', 'seat_support',
    'pedestal', 'support', 'lever', 'footrest', 'headrest',
    'vertical_back_panel', 'backrest_connector', 'wheel', 'stretcher',
    'armrest', 'backrest_frame', 'armrest_cushion', 'armrest_bar', 'rivet'
]

save_path = './part_name_embeddings'

# Generating a hypothetical corpus using these part names
# corpus = [
#     ['seat', 'attaches', 'to', 'seat_frame', 'with', 'seat_connector'],
#     ['seat_frame', 'connects', 'to', 'leg', 'via', 'rivet'],
#     ['backrest', 'mounts', 'on', 'seat_frame', 'using', 'backrest_connector'],
#     ['armrest', 'fixed', 'to', 'seat_frame', 'through', 'armrest_connector'],
#     ['footrest', 'links', 'to', 'lever', 'and', 'support', 'for', 'adjustment'],
#     ['headrest', 'attaches', 'to', 'backrest_frame', 'with', 'rivet'],
#     ['wheel', 'fits', 'into', 'pedestal', 'for', 'mobility'],
#     ['stretcher', 'connects', 'leg', 'to', 'leg', 'for', 'enhanced', 'stability'],
#     ['armrest', 'complemented', 'by', 'armrest_cushion', 'for', 'comfort'],
#     ['vertical_back_panel', 'assembles', 'with', 'backrest', 'and', 'backrest_frame'],
#     ['foot', 'used', 'for', 'stability', 'of', 'pedestal'],
#     ['pedestal', 'supports', 'seat_frame', 'and', 'includes', 'lever', 'for', 'height', 'adjustment'],
#     ['armrest_bar', 'connects', 'armrest', 'to', 'seat_frame'],
#     ['pillow', 'placed', 'on', 'seat', 'for', 'additional', 'comfort'],
#     ['lever', 'used', 'to', 'adjust', 'height', 'of', 'seat', 'and', 'backrest']
# ]
corpus = [
    ['seat', 'attaches', 'to', 'seat_frame', 'with', 'seat_connector'],
    ['seat_frame', 'connects', 'to', 'leg', 'via', 'rivet'],
    ['backrest', 'mounts', 'on', 'seat_frame', 'using', 'backrest_connector'],
    ['armrest', 'fixed', 'to', 'seat_frame', 'through', 'armrest_connector'],
    ['footrest', 'links', 'to', 'lever', 'and', 'support', 'for', 'adjustment'],
    ['headrest', 'attaches', 'to', 'backrest_frame', 'with', 'rivet'],
    ['wheel', 'fits', 'into', 'pedestal', 'for', 'mobility'],
    ['stretcher', 'connects', 'leg', 'to', 'leg', 'for', 'enhanced', 'stability'],
    ['armrest', 'complemented', 'by', 'armrest_cushion', 'for', 'comfort'],
    ['vertical_back_panel', 'assembles', 'with', 'backrest', 'and', 'backrest_frame'],
    ['foot', 'used', 'for', 'stability', 'of', 'pedestal'],
    ['pedestal', 'supports', 'seat_frame', 'and', 'includes', 'lever', 'for', 'height', 'adjustment'],
    ['armrest_bar', 'connects', 'armrest', 'to', 'seat_frame'],
    ['pillow', 'placed', 'on', 'seat', 'for', 'additional', 'comfort'],
    ['lever', 'used', 'to', 'adjust', 'height', 'of', 'seat', 'and', 'backrest'],
    ['seat_support', 'installed', 'under', 'seat_frame', 'to', 'provide', 'structural', 'support'],
    ['backrest_frame', 'supports', 'backrest', 'and', 'is', 'reinforced', 'by', 'vertical_back_panel'],
    ['rivet', 'used', 'to', 'secure', 'armrest_bar', 'and', 'armrest_connector']
]

# Train a Word2Vec Skip-Gram model(sg=1)
model = Word2Vec(sentences=corpus, vector_size=128, window=2, sample = 1e-3  # Threshold for configuring which higher-frequency words are randomly downsampled
,min_count=1, sg=1,epochs=10)

# Access vectors for a specific word
# word_vector = model.wv['backrest']  # Replace 'backrest' with any part name from your corpus
# print("Vector for 'backrest':", word_vector)

partNameEmbeddings = {}

for part in part_names:
    # print(f"Vector for '{part}':", model.wv[part])
    partNameEmbeddings[part] = model.wv[part].astype(np.float64)
    path = os.path.join(save_path, part)
    np.savez_compressed(path, model.wv[part])


# # Save the embeddings to a json file
# file_path = 'part_name_embeddings.json'
# with open(file_path, 'w') as f:
#     json.dump(partNameEmbeddings, f)


print("Embeddings saved !")
# only plot the part names embeddings



# Load embeddings from saved files
part_name_embeddings = {}
for part in part_names:
    path = os.path.join(save_path, part + '.npz')  # Adjust file extension if necessary
    with np.load(path) as data:
        part_name_embeddings[part] = data['arr_0']

# Create a matrix of all embeddings
embedding_matrix = np.array(list(part_name_embeddings.values()))


# Load embeddings from saved files, assuming they've already been loaded into part_name_embeddings as before
# embedding_matrix = np.array(list(part_name_embeddings.values()))

# Apply t-SNE to reduce dimensions to 2
# Check how many embeddings you have
n_samples = len(embedding_matrix)
print("Number of samples:", n_samples)
# Set perplexity to less than the number of samples, appropriately for the dataset size
perplexity_value = max(5, n_samples / 2)  # for example, half of the number of samples or 5

# Apply t-SNE with adjusted perplexity
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
reduced_embeddings = tsne.fit_transform(embedding_matrix)

# Plotting the embeddings
plt.figure(figsize=(10, 10))
for i, part in enumerate(part_names):
    plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], label=part)

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE of Part Names Embeddings')
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.03))
plt.grid(True)
plt.show()

# # Apply PCA to reduce dimensions to 2
# pca = PCA(n_components=2)
# reduced_embeddings = pca.fit_transform(embedding_matrix)
#
# # Plotting the embeddings
# plt.figure(figsize=(10, 10))
# for i, part in enumerate(part_names):
#     plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], label=part)
#
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.title('PCA of Part Names Embeddings')
# plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.03))
# plt.grid(True)
# plt.show()