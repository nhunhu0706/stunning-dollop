import streamlit as st
from PIL import Image
import requests
import numpy as np

st.title('Image Segmentation with kMeans')
col1, col2 = st.columns(2)
with col1:
    link = st.text_input('Image URL (Press Enter to apply)')
with col2:
    k = st.slider('K',2,10,3,1)
if link:
    
    def process(link,k):
        img_url = link
        img_pil = Image.open(requests.get(img_url, stream=True).raw)

        img = np.array(img_pil)

        X = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.fit(X)
        img_new = kmeans.cluster_centers_[kmeans.labels_]
        img_new = img_new.reshape(img.shape[0], img.shape[1], img.shape[2])
        img_new = img_new.astype(np.uint8)
        return img_new
    c1,c2 = st.columns(2)
    with c1:
        st.image(link)
        st.caption('Original Image')
    with c2:
        st.image(process(link,k))
        st.caption('Segmented Image')