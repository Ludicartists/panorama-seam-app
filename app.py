import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

EDGE_STRIP = 10

def load_rgb(file_obj):
    return np.array(Image.open(file_obj).convert('RGB'))

def edge_strips(img, edge_strip):
    left = img[:, :edge_strip, :]
    right = img[:, -edge_strip:, :]
    return left, right

def seam_score(left, right):
    diff = np.abs(left.astype(int) - right.astype(int))
    score = diff.mean()
    return score, diff

def log_result(path, score):
    st.write(f'Logged: {path} — seam score: {score:.3f}')

st.title('Seam Score & 95th Percentile Batch Analysis')

uploaded_files = st.file_uploader(
    "Upload images", type=["jpg","jpeg","png"], accept_multiple_files=True
)

if uploaded_files:
    for idx, file in enumerate(uploaded_files):
        # Load image
        img = load_rgb(file)
        fname = file.name

        if img.shape[1] < EDGE_STRIP*2:
            st.warning(f"{fname} too narrow for EDGE_STRIP={EDGE_STRIP}. Skipping.")
            continue

        # Edge strips
        left, right = edge_strips(img, EDGE_STRIP)

        # --- FIRST CODE BLOCK: seam score ---
        score, diff = seam_score(left, right)
        st.code(f'Image {idx+1} ({fname}) - Seam difference score: {score:.3f} (lower is better)')
        log_result(fname, score)

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].imshow(left)
        ax[0].set_title('Left edge')
        ax[0].axis('off')
        ax[1].imshow(right)
        ax[1].set_title('Right edge')
        ax[1].axis('off')
        ax[2].imshow(np.clip(diff.astype(np.uint8), 0, 255))
        ax[2].set_title('Difference heatmap')
        ax[2].axis('off')
        plt.suptitle(f'Image {idx+1}: {fname}')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        st.image(buf, caption='Notebook block 1: Edges & Diff Heatmap')

        # --- SECOND CODE BLOCK: 95th percentile ---
        diff2 = np.abs(left.astype(int) - right.astype(int))    # shape (H, W, 3)
        values = diff2.max(axis=2).flatten()                    # max-channel per pixel
        mean_diff = values.mean()
        p95_diff  = np.percentile(values, 95)
        max_diff  = values.max()
        st.code(f'Image {idx+1} ({fname}) → mean={mean_diff:.3f}, 95th%={p95_diff:.1f}, max={max_diff:.1f}')

        fig2, ax2 = plt.subplots(1, 3, figsize=(14, 4))
        ax2[0].imshow(left)
        ax2[0].set_title('Left edge')
        ax2[0].axis('off')
        ax2[1].imshow(right)
        ax2[1].set_title('Right edge')
        ax2[1].axis('off')
        ax2[2].imshow(diff2.max(axis=2), cmap='hot')
        ax2[2].set_title('Max-channel Difference Heatmap')
        ax2[2].axis('off')
        plt.suptitle(f'Image {idx+1}: {fname}')
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', bbox_inches='tight')
        plt.close(fig2)
        buf2.seek(0)
        st.image(buf2, caption='Notebook block 2: Max-channel Difference Heatmap')
