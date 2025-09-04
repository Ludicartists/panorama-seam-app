import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from fpdf import FPDF

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

def pil_bytes(img_array):
    pil_img = Image.fromarray(np.uint8(img_array))
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return buf

st.title('Seam Score & 95th Percentile Batch Analysis')
uploaded_files = st.file_uploader(
    "Upload images", type=["jpg","jpeg","png"], accept_multiple_files=True
)

# Store summary info for the PDF report
report_images = []
report_texts = []

if uploaded_files:
    for idx, file in enumerate(uploaded_files):
        img = load_rgb(file)
        fname = file.name
        if img.shape[1] < EDGE_STRIP*2:
            st.warning(f"{fname} too narrow for EDGE_STRIP={EDGE_STRIP}. Skipping.")
            continue
        left, right = edge_strips(img, EDGE_STRIP)
        score, diff = seam_score(left, right)
        st.code(f'Image {idx+1} ({fname}) - Seam difference score: {score:.3f} (lower is better)')
        log_result(fname, score)
        # Figure 1: Standard difference heatmap
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
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png', bbox_inches='tight')
        plt.close(fig)
        buf1.seek(0)
        st.image(buf1, caption='Notebook block 1: Edges & Diff Heatmap')
        diff2 = np.abs(left.astype(int) - right.astype(int))
        values = diff2.max(axis=2).flatten()
        mean_diff = values.mean()
        p95_diff  = np.percentile(values, 95)
        max_diff  = values.max()
        st.code(f'Image {idx+1} ({fname}) → mean={mean_diff:.3f}, 95th%={p95_diff:.1f}, max={max_diff:.1f}')
        # Figure 2: Max-channel Heatmap
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
        # For PDF: Store (image, plots, summary)
        img_buf = pil_bytes(img)
        report_images.append(dict(
            fname=fname,
            img=img_buf,
            heatmap1=buf1,
            heatmap2=buf2
        ))
        text = (
            f"Image {idx+1}: {fname}\n"
            f"Seam score: {score:.3f}\n"
            f"Mean diff: {mean_diff:.3f}\n"
            f"95th percentile: {p95_diff:.1f}\n"
            f"Max diff: {max_diff:.1f}"
        )
        report_texts.append(text)

    # --- GENERATE PDF BUTTON: SHOWS ONLY AFTER UPLOAD ---
    if st.button("Generate PDF Summary Report"):
        pdf = FPDF()
        for i, (imgs, text) in enumerate(zip(report_images, report_texts)):
            pdf.add_page()
            # Section header - centered
            pdf.set_font("Arial", 'B', 16)
            pdf.set_text_color(22, 86, 165)
            pdf.cell(0, 12, f"Image {i+1}: {imgs['fname']}", ln=True, align='C')
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)
            # Show the original image big and centered at top
            img_data = imgs['img'].getvalue()
            with open(f"img_{i}.png", "wb") as f: f.write(img_data)
            orig_w = 70
            img_x = (210 - orig_w) // 2
            img_y = pdf.get_y()
            pdf.image(f"img_{i}.png", x=img_x, y=img_y, w=orig_w)
            pdf.ln(orig_w + 3)
            # Smaller, centered stats table
            text_lines = text.split("\n")[1:]  # skip redundant filename
            col1 = ["Seam", "Mean", "95th %", "Max"]
            col2 = [l.split(": ")[1] for l in text_lines]
            table_w = 30
            value_w = 30
            table_start_x = (210 - (table_w + value_w)) // 2
            pdf.set_xy(table_start_x, pdf.get_y())
            pdf.set_font("Arial", 'B', 11)
            pdf.set_fill_color(22, 86, 165)
            pdf.set_text_color(255,255,255)
            pdf.cell(table_w, 8, "Metric", border=1, align="C", fill=True)
            pdf.cell(value_w, 8, "Value", border=1, align="C", fill=True)
            pdf.ln(8)
            pdf.set_font("Arial", '', 11)
            pdf.set_text_color(0,0,0)
            pdf.set_x(table_start_x)
            for c1, c2 in zip(col1, col2):
                pdf.cell(table_w, 8, c1, border=1, align='C')
                pdf.cell(value_w, 8, c2, border=1, align='C')
                pdf.ln(8)
                pdf.set_x(table_start_x)
            pdf.ln(5)
            # Divider line
            y = pdf.get_y()
            pdf.set_draw_color(180, 180, 180)
            pdf.set_line_width(0.5)
            pdf.line(20, y, 190, y)
            pdf.ln(5)
            # Large Heatmaps, side by side, centered
            heatmap1_data = imgs['heatmap1'].getvalue()
            heatmap2_data = imgs['heatmap2'].getvalue()
            with open(f"hm1_{i}.png", "wb") as f: f.write(heatmap1_data)
            with open(f"hm2_{i}.png", "wb") as f: f.write(heatmap2_data)
            big_w = 80
            margin = (210 - 2*big_w) // 3
            y0 = pdf.get_y()
            x1 = margin
            x2 = margin * 2 + big_w
            pdf.image(f"hm1_{i}.png", x=x1, y=y0+3, w=big_w)
            pdf.image(f"hm2_{i}.png", x=x2, y=y0+3, w=big_w)
            # Captions under each heatmap
            pdf.set_font("Arial", 'I', 10)
            pdf.set_text_color(22,86,165)
            cap_y = y0 + big_w + 8
            pdf.set_xy(x1, cap_y)
            pdf.cell(big_w, 8, "Edge & Diff Heatmap", align="C")
            pdf.set_xy(x2, cap_y)
            pdf.cell(big_w, 8, "Max-Channel Difference", align="C")
            pdf.set_y(cap_y+14)
            # Footer page number
            pdf.set_text_color(120,120,120)
            pdf.set_font("Arial", 'I', 9)
            pdf.cell(0, 10, f"Page {pdf.page_no()}", align='C')
        pdf_output = pdf.output(dest="S").encode("latin-1")
        st.download_button(
            label="Download PDF Report",
            data=pdf_output,
            file_name="Seam_Score_Report.pdf",
            mime="application/pdf"
        )
