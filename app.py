import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import zipfile, io, gc
from io import BytesIO

# ---------- CONSTANTS ----------
FIGURE_SIZE = 7.5
EDITION = "2023"
TITLE_MAP = {"EN": "Sample Code", "FR": "Echantillon", "ES": "Muestra"}

EXPECTED_HEADERS = [
    "Master_code", "Global_Quality", "Cacao", "Acid_Total", "Acid_Fr", "Acid_Ac", "Acid_Lac", "Acid_MB",
    "Bitterness", "Astringency", "FFruit_Total", "FFruit_Be", "FFruit_Ci", "FFruit_Da", "FFruit_YOW", "FFruit_Tr",
    "BFruit_Total", "BFruit_Dr", "BFruit_Br", "BFruit_Ov", "Vegetal_Total", "Vegetal_Gr", "Vegetal_Ea",
    "Floral_Total", "Floral_OB", "Floral_Fl", "Wood_Total", "Wood_Li", "Wood_Da", "Wood_Re",
    "Spice_Total", "Spice_Sp", "Spice_To", "Spice_Sa",
    "Nutty_Total", "Nutty_Fl", "Nutty_Sk", "Panela", "Sweetness", "Roast",
    "OF_Total", "OF_Dirty", "OF_Musty", "OF_Mouldy", "OF_Meaty", "OF_Over_ferm",
    "OF_Putrid", "OF_Smoky", "OF_Other", "Other_OF_Description", "Overall_Flavour_Comment", "Feedback_Comment"
]

CACAO_ATTRS = [
    'Cacao', 'Acid_Total', 'Bitterness', 'Astringency', 'FFruit_Total',
    'BFruit_Total', 'Vegetal_Total', 'Floral_Total', 'Wood_Total',
    'Spice_Total', 'Nutty_Total', 'Panela', 'OF_Total', 'Roast'
]
CHOC_ATTRS = CACAO_ATTRS[:-2] + ['Sweetness'] + CACAO_ATTRS[-2:]

CACAO_COLORS = [
    '#754C29', '#00954C', '#A01F65', '#366D99', '#F6D809', '#431614',
    '#006260', '#8DC63F', '#A97C50', '#C33D32', '#A0A368', '#BD7844',
    '#A7A9AC', '#EBAB21'
]
CHOC_COLORS = CACAO_COLORS[:-3] + ['#FFC6E0'] + CACAO_COLORS[-3:]  # pink for Sweetness

# ---------- UI ----------
st.set_page_config(page_title="Flavour Graph Generator", layout="centered")
st.image("masks/logo.png", width=120)
st.title("Flavour Graph Generator")

lang = st.selectbox("Select language", ["EN", "FR", "ES"])
eval_type = st.radio("Sample type", ["Cacao Mass", "Chocolate"])

# Downloadable template
def generate_template():
    return pd.DataFrame(columns=EXPECTED_HEADERS)

@st.cache_data
def get_template_bytes():
    df = generate_template()
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

st.download_button(
    label="📥 Download Excel Template",
    data=get_template_bytes(),
    file_name="flavour_graph_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

uploaded = st.file_uploader("Upload sensory evaluation Excel (.xlsx)", type=["xlsx", "xlsm"])
download_placeholder = st.empty()

# ---------- GRAPH GENERATOR ----------
def generate_zip(df, lang, eval_type):
    if eval_type == "Cacao Mass":
        attrs, colors, num_attrs, title_sub = CACAO_ATTRS, CACAO_COLORS, 14, "M"
    else:
        attrs, colors, num_attrs, title_sub = CHOC_ATTRS, CHOC_COLORS, 15, "C"

    try:
        mask_img = plt.imread(f"masks/{num_attrs}-Flavour-Wheel-MASK-{lang}.png")
    except FileNotFoundError:
        st.error("There was an issue.")
        return None

    theta = np.radians(np.linspace(360 - 360/len(attrs), 0, len(attrs)))
    width = np.radians(360 / len(attrs))

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
        try:
            for code in df.index:
                radii = [df.loc[code, col] for col in attrs]

                fig = plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))
                ax_mask = fig.add_axes([0, 0, 1, 1])           # background/mask axes
                ax_mask.imshow(mask_img)
                ax_mask.axis("off")

                # Polar chart axes
                ax = fig.add_axes([0.01, 0.01, 0.98, 0.98],
                                  projection="polar",
                                  theta_offset=np.radians(90),
                                  aspect=1)
                ax.patch.set_alpha(0)
                ax.set_ylim(0, 10)
                ax.set_xticks(theta)
                ax.set_xticklabels([])
                ax.bar(theta, radii, width=width, bottom=0.0, color=colors, align="edge")

                # radial grid labels
                for label in [2, 4, 6, 8, 10]:
                    t = plt.text(0, label, str(label), ha="center", va="center", size=10)
                    t.set_path_effects([PathEffects.withStroke(linewidth=5, foreground="w")])
                ax.set_rgrids([])
                ax.spines["polar"].set_visible(False)

                # NEW: Title drawn on the outer mask axes, aligned hard-left and with no extra indentation
                title_str = f"{TITLE_MAP[lang]}\n{code} {title_sub}"
                txt = ax_mask.text(
                    0.012, 0.975,                 # further left & top (figure fraction)
                    title_str,
                    transform=ax_mask.transAxes,
                    ha="left", va="top",
                    fontsize=15, weight="bold"
                )
                # Outline for contrast over mask
                txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground="w")])

                # Save to in-memory PNG then into the ZIP
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                z.writestr(f"{code} - {eval_type} Graph {lang}.png", buf.read())
                plt.close(fig)
                gc.collect()
        except Exception:
            st.error("There was an issue.")
            return None

    zip_buffer.seek(0)
    return zip_buffer

# ---------- MAIN ----------
if uploaded:
    try:
        df = pd.read_excel(uploaded, engine='openpyxl')

        if list(df.columns) == EXPECTED_HEADERS:
            # NEW: Validate Master_code length (<= 10). Also normalize to string & strip spaces first.
            df["Master_code"] = df["Master_code"].astype(str).str.strip()
            too_long = df.loc[df["Master_code"].str.len() > 10, "Master_code"].unique().tolist()
            if too_long:
                st.error("❌ Some Master_code values exceed the 10-character limit. Please shorten them and re-upload.")
                st.write("Offending codes (first 20):", too_long[:20])
                st.stop()

            df.set_index("Master_code", inplace=True)

            if st.button("Download flavour graphs"):
                zip_file = generate_zip(df, lang, eval_type)
                if zip_file:
                    download_placeholder.download_button(
                        "Click here to download ZIP file",
                        data=zip_file,
                        file_name=f"{eval_type.replace(' ', '_')}_Graphs_{lang}.zip",
                        mime="application/zip"
                    )
        else:
            st.error("❌ Uploaded file does not match the expected template structure.")
            st.write("Expected headers:")
            st.write(EXPECTED_HEADERS)
            st.write("Uploaded file headers:")
            st.write(list(df.columns))

    except Exception as e:
        st.error(f"There was an issue reading the file: {e}")
