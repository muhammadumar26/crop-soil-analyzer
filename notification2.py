import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import firebase_admin
from firebase_admin import credentials, storage, db
import os
import urllib.parse


firebase_config = {
    "type": st.secrets["type"],
    "project_id": st.secrets["project_id"],
    "private_key_id": st.secrets["private_key_id"],
    "private_key": st.secrets["private_key"].replace("\\n", "\n"),
    "client_email": st.secrets["client_email"],
    "client_id": st.secrets["client_id"],
    "auth_uri": st.secrets["auth_uri"],
    "token_uri": st.secrets["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["client_x509_cert_url"],
    "universe_domain": st.secrets.get("universe_domain", "googleapis.com") # Optional fallback
}

cred = credentials.Certificate(firebase_config)

# Firebase Setup
APP_NAME = 'analysis_app'

try:
    app = firebase_admin.get_app(APP_NAME)
except ValueError:
    app = firebase_admin.initialize_app(
        cred,
        {
            'storageBucket': 'agrobotix-d23e1',  # your bucket name
            'databaseURL': 'https://agrobotix-d23e1-default-rtdb.europe-west1.firebasedatabase.app/'
        },
        name=APP_NAME
    )

bucket = storage.bucket(app=app)
ref = db.reference('reports', app=app)

# Streamlit UI Setup
plt.switch_backend('Agg')
plt.style.use('dark_background')
st.set_page_config(page_title="🌱/🌾 Crop & Soil Analyzer", layout="wide")
st.title("🌿 Crop & Soil Health Analyzer")
st.markdown("Choose analysis type and upload an image to analyze vegetation or soil health.")

st.sidebar.image("team_logo.jpg", use_container_width=True)
st.sidebar.markdown("# **Team EMU AGROBOTIX**")
st.sidebar.markdown("---")
st.sidebar.image("slogan.jpg", use_container_width=True)
st.sidebar.image("horizonx.jpg", use_container_width=True)
st.sidebar.image("slogan2.jpg", use_container_width=True)
st.sidebar.image("emu.jpg", use_container_width=True)
st.sidebar.markdown("---")

# Functions
@st.cache_data
def extract_gps_coordinates_pillow(file):
    try:
        img = Image.open(file)
        exif_data = img._getexif()
        if not exif_data:
            return "📍 Location: No EXIF data found"
        gps_info = {}
        for tag, value in exif_data.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_info[sub_decoded] = value[t]
        if 'GPSLatitude' not in gps_info:
            return "📍 Location: No GPS data found"
        def _conv(rat):
            try:
                return float(rat.numerator)/rat.denominator
            except:
                return 0.0
        def _to_deg(vals):
            d = _conv(vals[0]); m = _conv(vals[1]); s = _conv(vals[2]) if len(vals)>2 else 0
            return d + m/60 + s/3600
        lat = _to_deg(gps_info['GPSLatitude'])
        if gps_info.get('GPSLatitudeRef','N')=='S': lat = -lat
        lon = _to_deg(gps_info['GPSLongitude'])
        if gps_info.get('GPSLongitudeRef','E')=='W': lon = -lon
        return f"📍 GPS Coordinates: {lat:.6f}°, {lon:.6f}°"
    except Exception as e:
        return f"📍 Location Error: {str(e)}"

@st.cache_data
def calculate_vegetation_indices(img):
    img = img.astype(np.float32)/255.0
    b,g,r = cv2.split(img)
    eps = 1e-10
    return {
        'NDVI': (g-r)/(g+r+eps),
        'GLI': (2*g - r - b)/(2*g + r + b + eps),
        'VARI': (g-r)/(g+r-b+eps),
        'ExG': (2*g - r - b),
        'GRVI': (g-r)/(g+r+eps),
        'NGRDI': (g-r)/(g+r+eps),
        'RGBVI': (g**2 - r*b)/(g**2 + r*b + eps)
    }

@st.cache_data
def calculate_soil_indices(img):
    img = img.astype(np.float32)/255.0
    b,g,r = cv2.split(img)
    eps = 1e-10
    return {
        'Brightness': np.sqrt((r**2+g**2+b**2)/3),
        'NDI': (r-g)/(r+g+eps),
        'ColorIndex': (r-b)/(r+b+eps),
        'ExG': (2*g - r - b),
        'VARI': (g-r)/(g+r-b+eps),
        'VDI': (2*b - r - g)/(2*b + r + g + eps)
    }

def apply_vegetation_colormap(index):
    defs = {
        'NDVI': ['#b2182b','#ef8a62','#fddbc7','#a6dba0','#1b7837'],
        'GLI': ['#ffffcc','#d9f0a3','#7fbc41','#4d9221','#276419'],
        'VARI': ['#8c510a','#d8b365','#f5f5f5','#58d68d','#52be80'],
        'ExG': ['#404040','#878787','#cccccc','#b8e186','#4d9221'],
        'GRVI': ['#67001f','#b2182b','#f4a582','#a6dba0','#1b7837'],
        'NGRDI': ['#1b7837','#52be80','#58d68d','#abebc6','#f0f0f0'],
        'RGBVI':['#762a83','#c51b7d','#de77ae','#b8e186','#4d9221']
    }
    return LinearSegmentedColormap.from_list(f"veg_{index}", defs[index])

def apply_soil_colormap(index):
    cmap_defs = {
        'Brightness':['#2c7bb6','#abd9e9','#ffffbf','#fdae61','#d7191c'],
        'NDI':['#313695','#4575b4','#74add1','#fdae61','#d73027'],
        'ColorIndex': ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c'],
        'ExG': ['#404040','#878787','#cccccc','#b8e186','#4d9221'],
        'VARI': ['#8c510a','#d8b365','#f5f5f5','#58d68d','#52be80'],
        'VDI': ['#67001f', '#b2182b', '#ef8a62', '#fddbc7', '#f7f7f7']
    }
    return LinearSegmentedColormap.from_list(f"soil_{index}", cmap_defs[index])
def add_veg_description(ax, idx):
    desc = {
        'NDVI': 'Normalized Difference Veg Index: red shows low/no veg; green = healthy.',
        'GLI': 'Green Leaf Index: yellow→green indicate health.',
        'VARI': 'VARI: brown low veg; green high veg.',
        'ExG': 'Excess Green: gray→green = more vegetation.',
        'GRVI': 'Green-Red Veg Index: red = low veg; green = healthy.',
        'NGRDI': 'Norm Green-Red Diff: green = healthy veg.',
        'RGBVI': 'RGB Veg Index: purple = unhealthy; green = healthy.'
    }
    ax.text(0.05, 0.95, desc[idx], transform=ax.transAxes, fontsize=7,
            va='top', bbox=dict(facecolor='black', alpha=0.7, edgecolor='white', boxstyle='round,pad=0.3'))

def add_soil_description(ax, idx):
    desc = {
        'Brightness': 'Brightness: blue=dark soil; red=sandy bright soil.',
        'NDI': 'NDI: red>blue indicates iron content.',
        'ColorIndex': 'ColorIndex: green=wet; red=oxidized dry soil.',
        'ExG': 'ExG: green indicates vegetation on soil.',
        'VARI': 'VARI: resists atmosphere, highlights green veg.',
        'VDI': 'VDI: red=dry; white=balanced moisture.'
    }
    ax.text(0.05, 0.95, desc[idx], transform=ax.transAxes, fontsize=7,
            va='top', bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

def generate_veg_report(indices, fname, gps):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report = []
    report.append(f"=== 🌿 Vegetation Analysis Report ===")
    report.append(f"File: {fname}")
    report.append(gps)
    report.append(f"Date: {now}\n")

    # Overall NDVI assessment
    ndvi_mean = np.mean(indices['NDVI'])
    report.append(f"1. NDVI Mean: {ndvi_mean:.3f}")
    if ndvi_mean > 0.6:
        report.append("   - Excellent vegetation density and health.")
        report.append("     * Continue current agronomic practices.")
        report.append("     * Monitor for potential overgrowth or pests.")
    elif ndvi_mean > 0.3:
        report.append("   - Good vegetation health.")
        report.append("     * Check soil fertility and irrigation.")
    elif ndvi_mean > 0.1:
        report.append("   - Moderate vegetation. Possible early stress.")
        report.append("     * Recommend soil and moisture monitoring.")
    else:
        report.append("   - Poor vegetation. Urgent intervention needed.")
        report.append("     * Soil amendment and pest control required.")

    # Per-index breakdown with interpretations
    thresholds = {
        'GLI': (0.2, 0.6),
        'ExG': (0.1, 0.5),
        'VARI': (0.02, 0.3),
        'GRVI': (0.1, 0.5),
        'NGRDI': (0.1, 0.5),
        'RGBVI': (0.1, 0.6)
    }
    descriptions = {
        'GLI': 'Green Leaf Index: higher indicates more healthy foliage.',
        'ExG': 'Excess Green: highlights green coverage over background.',
        'VARI': 'Vegetation Atmos Res Index: compensates for atmospheric effects.',
        'GRVI': 'Green-Red Veg Index: simple greenness vs redness metric.',
        'NGRDI': 'Norm Green-Red Diff: sensitive to chlorophyll content.',
        'RGBVI': 'RGB Veg Index: suppresses red/blue noise, emphasizes green.'
    }
    report.append("\n2. Detailed Indices Breakdown:")
    for name, (low, high) in thresholds.items():
        mean_val = np.mean(indices[name])
        status = '✅ Healthy' if low <= mean_val <= high else '⚠️ Attention'
        suggestion = ''
        if mean_val < low:
            suggestion = 'Consider nutrient supplementation or irrigation.'
        elif mean_val > high:
            suggestion = 'Ensure adequate spacing to prevent overcrowding.'
        report.append(f"   - {name}: {mean_val:.3f} ({status})")
        report.append(f"     * {descriptions[name]}")
        report.append(f"     * Suggestion: {suggestion}")

    report.append("\nNote: Index thresholds vary by crop and environment. Adjust as needed.")
    return "\n".join(report)


def generate_soil_report(indices, fname, gps):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report = []
    report.append(f"=== 🌾 Soil Analysis Report ===")
    report.append(f"File: {fname}")
    report.append(gps)
    report.append(f"Date: {now}\n")

    # Brightness assessment
    bright = np.mean(indices['Brightness'])
    report.append(f"1. Soil Brightness Mean: {bright:.3f}")
    if bright > 0.6:
        report.append("   - High brightness: sandy or low organic matter.")
        report.append("     * Add compost and organic mulch.")
    elif bright < 0.3:
        report.append("   - Low brightness: rich in organic matter or moisture.")
        report.append("     * Good water retention; monitor for waterlogging.")
    else:
        report.append("   - Moderate brightness: balanced loamy soil.")

    # Moisture (VDI)
    vdi = np.mean(indices['VDI'])
    report.append(f"\n2. Soil Moisture (VDI) Mean: {vdi:.3f}")
    if vdi > 0.2:
        report.append("   - Dry soil detected.")
        report.append("     * Immediate irrigation recommended.")
    elif vdi < -0.2:
        report.append("   - Wet soil detected.")
        report.append("     * Improve drainage to prevent root rot.")
    else:
        report.append("   - Normal moisture levels.")

    # Vegetation presence on soil
    exg = np.mean(indices['ExG'])
    report.append(f"\n3. Vegetation Cover (ExG) Mean: {exg:.3f}")
    if exg > 0.2:
        report.append("   - Significant green cover: alive vegetation present.")
        report.append("     * Monitor nutrient needs for healthy growth.")
    elif exg < -0.2:
        report.append("   - Bare soil detected.")
        report.append("     * Consider cover crops or mulching.")
    else:
        report.append("   - Low to moderate vegetation cover.")

    # Oxidation & iron (ColorIndex & NDI)
    ci = np.mean(indices['ColorIndex'])
    ndi = np.mean(indices['NDI'])
    report.append(f"\n4. Soil Oxidation (ColorIndex) Mean: {ci:.3f}")
    report.append(ci > 0.15 and "   - Well-aerated (oxidized)." or "   - Reduced/waterlogged conditions.")
    report.append(f"5. Iron Content (NDI) Mean: {ndi:.3f}")
    report.append(ndi > 0.1 and "   - High iron content." or "   - Lower iron levels; consider supplements." )

    # Additional indexes
    report.append("\n6. Additional Indices:")
    report.append(f"   - VARI: {np.mean(indices['VARI']):.3f} (atmospherically resistant index)")
    report.append(f"   - NGRDI: {np.mean(indices.get('NGRDI',indices['VARI'])):.3f} (normalized green-red diff)")

    report.append("\n=== Recommendations ===")
    recommendations = []
    if bright > 0.6 and vdi > 0.2:
        recommendations.append("Add organic matter and irrigate regularly.")
    if bright < 0.3 and vdi < -0.2:
        recommendations.append("Improve soil drainage.")
    if exg < -0.2:
        recommendations.append("Plant cover crops or mulch.")
    if not recommendations:
        recommendations.append("Soil conditions are within optimal ranges.")
    for r in recommendations:
        report.append(f"- {r}")

    report.append("\nNote: Combine with laboratory soil tests for best accuracy.")

    return "\n".join(report)

# Main
def main():
    analysis_type = st.selectbox("Select Analysis", ["Vegetation Analysis", "Soil Analysis"])
    uploaded = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

    if uploaded:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        uploaded.seek(0)
        gps = extract_gps_coordinates_pillow(uploaded)

        if analysis_type == "Vegetation Analysis":
            indices = calculate_vegetation_indices(img)
            report = generate_veg_report(indices, uploaded.name, gps)
        else:
            indices = calculate_soil_indices(img)
            report = generate_soil_report(indices, uploaded.name, gps)

        # Upload original image
        image_blob = bucket.blob(f"uploaded_images/{uploaded.name}_{ts}")
        uploaded.seek(0)
        image_blob.upload_from_file(uploaded, content_type="image/jpeg")

        bucket_name = image_blob.bucket.name   # <-- Corrected
        blob_name = image_blob.name             # <-- Corrected
        blob_name_encoded = urllib.parse.quote(blob_name, safe="")
        image_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{blob_name_encoded}?alt=media"


        # Show layout
        col1, col2 = st.columns([1,2])
        with col1:
            st.image(img_rgb, caption="Original Image", use_container_width=True)
        with col2:
            tabs = st.tabs(list(indices.keys()))
            index_urls = {}
            for tab, (name, mat) in zip(tabs, indices.items()):
                with tab:
                    fig, ax = plt.subplots(figsize=(6,6))
                    cmap = apply_vegetation_colormap(name) if "Veg" in analysis_type else apply_soil_colormap(name)
                    norm = plt.Normalize(np.nanpercentile(mat,5), np.nanpercentile(mat,95))
                    im = ax.imshow(mat, cmap=cmap, norm=norm)
                    plt.colorbar(im, ax=ax)
                    if "Veg" in analysis_type:
                        add_veg_description(ax, name)
                    else:
                        add_soil_description(ax, name)
                    ax.axis('off')
                    st.pyplot(fig)

                    # Save and upload each index image
                    fig_path = f"{name}_{ts}.png"
                    fig.savefig(fig_path, bbox_inches='tight')
                    blob = bucket.blob(f"indices/{uploaded.name}_{ts}/{name}.png")
                    blob.upload_from_filename(fig_path)
                    bucket_name = blob.bucket.name
                    blob_name = blob.name
                    blob_name_encoded = urllib.parse.quote(blob_name, safe="")
                    firebase_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{blob_name_encoded}?alt=media"
                    index_urls[name] = firebase_url
                    plt.close(fig)

        st.subheader("📄 Report Summary")
        st.text(report)

        # Push to Realtime Database
# Create the report entry with index URLs as separate fields
        report_entry = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': analysis_type,
            'image_name': uploaded.name,
            'image_url': image_url,  # Changed from image_urls to image_url for consistency
            'gps_location': gps,
            'report_text': report
        }

        # Add each index URL as a separate field
        for name, url in index_urls.items():
            report_entry[name] = url  # This will create fields like 'NDVI', 'GLI', etc.

        ref.push(report_entry)

        st.success("✅ Report and images successfully uploaded!")
        st.info(f"🌐 Original Image URL: {image_url}")

if __name__ == "__main__":
    main()
