import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import io
import zipfile

# Set page configuration
st.set_page_config(page_title="PDF/Image Footer Cropper", layout="wide")

def detect_header_line(image_np):
    """
    Attempts to detect the header line automatically.
    Returns the y-coordinate of the suggested cut line (top).
    """
    if image_np is None:
        return 0
        
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    # Binarize
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological opening to isolate horizontal lines
    line_min_width = width // 6
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min_width, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    potential_lines = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        line_center_y = y + h // 2
        
        # Filter for typical header position (top 20% of page)
        if 0 < line_center_y < 0.20 * height:
            potential_lines.append(line_center_y)
            
    if potential_lines:
        potential_lines.sort()
        return potential_lines[-1] # Return the lowest line in the top region (largest Y in that region)
    
    # Default fallback: 0 (no crop) or small margin
    return 0

def detect_footer_line(image_np):
    """
    Attempts to detect the footer line automatically.
    Returns the y-coordinate of the suggested cut line.
    """
    if image_np is None:
        return None
        
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    # Binarize
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological opening to isolate horizontal lines
    line_min_width = width // 6
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min_width, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    potential_lines = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        line_center_y = y + h // 2
        
        # Filter for typical footer position (bottom 35% of page)
        if 0.65 * height < line_center_y < 0.98 * height:
            potential_lines.append(line_center_y)
            
    if potential_lines:
        potential_lines.sort()
        return potential_lines[0] # Return the highest line (smallest Y)
    
    # Default fallback: 90% of height
    return int(0.90 * height)

def main():
    st.title("✂️ Batch Header & Footer Cropper")
    st.markdown("""
    1. **Upload** your files.
    2. **Review** each image and adjust the crop lines (Green = Header, Red = Footer).
    3. **Apply** settings to all if needed.
    4. **Download** the result.
    """)

    # --- Sidebar Settings ---
    with st.sidebar:
        st.header("Upload Settings")
        # output_format removed as users requested keeping original type
        st.info("Output images will have same name and format as original.")

    # --- Session State Initialization ---
    if 'images_data' not in st.session_state:
        st.session_state.images_data = [] # List of dicts: {'name', 'image', 'orig_name'}
    if 'crop_positions' not in st.session_state:
        st.session_state.crop_positions = {} # Map: name -> {'top': int, 'bottom': int}
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    # --- File Uploader ---
    uploaded_files = st.file_uploader("Choose files (PDF or Images)", type=['pdf', 'png', 'jpg', 'jpeg'], accept_multiple_files=True)

    # Process uploads
    if uploaded_files:
        if len(st.session_state.images_data) == 0:
            with st.spinner("Processing files..."):
                new_images = []
                for uploaded_file in uploaded_files:
                    file_type = uploaded_file.type
                    file_name = uploaded_file.name
                    
                    if "pdf" in file_type:
                        # PDF to Images
                        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                            for page_num in range(len(doc)):
                                page = doc[page_num]
                                pix = page.get_pixmap(dpi=200) # Balanced DPI
                                img_data = pix.tobytes("png")
                                img = Image.open(io.BytesIO(img_data))
                                new_images.append({
                                    "name": f"{file_name}_p{page_num+1}",
                                    "image": img,
                                    "orig_name": file_name
                                })
                    else:
                        # Standard Image
                        img = Image.open(uploaded_file)
                        new_images.append({
                            "name": file_name,
                            "image": img,
                            "orig_name": file_name
                        })
                
                st.session_state.images_data = new_images
                
                # Initial detection for all
                for item in new_images:
                    name = item['name']
                    img_np = np.array(item['image'])
                    
                    # Store as dict
                    st.session_state.crop_positions[name] = {
                        'top': detect_header_line(img_np),
                        'bottom': detect_footer_line(img_np)
                    }

    # --- Main Interface ---
    if not st.session_state.images_data:
        st.info("Upload files to start.")
        return

    # Get Current Image Data
    total_images = len(st.session_state.images_data)
    current_idx = st.session_state.current_index
    current_item = st.session_state.images_data[current_idx]
    current_name = current_item['name']
    current_pil = current_item['image']
    current_np = np.array(current_pil)
    height, width = current_np.shape[:2]

    # Helper for Navigation
    def render_navigation(key_prefix, total_images):
        col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1], gap="small")
        
        with col_nav1:
            if st.button("⬅️ Previous", key=f"{key_prefix}_prev", use_container_width=True):
                st.session_state.current_index = max(0, st.session_state.current_index - 1)
                st.rerun()
                
        with col_nav2:
            st.markdown(f"<p style='text-align: center; padding-top: 10px;'><b>Image {st.session_state.current_index + 1} of {total_images}</b></p>", unsafe_allow_html=True)
            
        with col_nav3:
            if st.button("Next ➡️", key=f"{key_prefix}_next", use_container_width=True):
                st.session_state.current_index = min(total_images - 1, st.session_state.current_index + 1)
                st.rerun()

    # --- Header Information ---
    st.divider()
    st.subheader(f"Edit: {current_name} ({current_idx + 1}/{total_images})")

    # --- Logic for Line Positions ---
    # Migration/Fallback check: ensure valid dict structure
    saved_pos = st.session_state.crop_positions.get(current_name, {})
    if isinstance(saved_pos, int): # Migration from old version if accidentally persisting
        saved_pos = {'top': 0, 'bottom': saved_pos}
    
    current_top = saved_pos.get('top', 0)
    current_bottom = saved_pos.get('bottom', height)

    slider_key_top = f"slider_top_{current_name}"
    slider_key_bottom = f"slider_bottom_{current_name}"
    
    # Priority: Slider -> State -> Default
    if slider_key_top in st.session_state:
        current_top = st.session_state[slider_key_top]
    
    if slider_key_bottom in st.session_state:
        current_bottom = st.session_state[slider_key_bottom]

    # ==========================
    # === TOP SECTION: HEADER ===
    # ==========================
    render_navigation("top", total_images)
    
    col_top_ctrl, col_top_batch = st.columns([3, 1], gap="medium")
    
    with col_top_ctrl:
        new_top = st.slider(
            "⬆️ Header Line (Top - Green)",
            min_value=0,
            max_value=height,
            value=int(current_top),
            key=slider_key_top
        )
        
    with col_top_batch:
        st.write("") # Spacer
        st.write("") # Spacer
        if st.button("Apply Top to ALL", key="batch_top"):
            count = 0
            for item in st.session_state.images_data:
                # Update top only
                current_vals = st.session_state.crop_positions.get(item['name'], {'top': 0, 'bottom': item['image'].height})
                if isinstance(current_vals, int): current_vals = {'top': 0, 'bottom': current_vals}
                
                current_vals['top'] = new_top
                st.session_state.crop_positions[item['name']] = current_vals
                
                # Update sliders if they exist
                k_top = f"slider_top_{item['name']}"
                if k_top in st.session_state and k_top != slider_key_top:
                    st.session_state[k_top] = new_top
                count += 1
            st.success(f"Applied Top={new_top} to {count} images.")
            st.rerun()

    st.divider()

    # ==========================
    # === MIDDLE: PREVIEW    ===
    # ==========================
    preview_img = current_np.copy()
    
    # Draw Lines
    # Green for Header (Top)
    cv2.line(preview_img, (0, int(new_top)), (width, int(new_top)), (0, 255, 0), max(1, height//200))
    # Red for Footer (Bottom) - Note we use current_bottom here initially, updated below
    cv2.line(preview_img, (0, int(current_bottom)), (width, int(current_bottom)), (255, 0, 0), max(1, height//200)) 
    
    st.image(preview_img, use_container_width=True, caption=f"Preview: {current_name}")

    st.divider()

    # ==========================
    # === BOTTOM SECTION: FOOTER ===
    # ==========================
    
    col_bottom_ctrl, col_bottom_batch = st.columns([3, 1], gap="medium")
    
    with col_bottom_ctrl:
        new_bottom = st.slider(
            "⬇️ Footer Line (Bottom - Red)",
            min_value=0,
            max_value=height,
            value=int(current_bottom),
            key=slider_key_bottom
        )
        
    with col_bottom_batch:
        st.write("")
        st.write("")
        if st.button("Apply Bottom to ALL", key="batch_bottom"):
            count = 0
            for item in st.session_state.images_data:
                # Update bottom only
                current_vals = st.session_state.crop_positions.get(item['name'], {'top': 0, 'bottom': item['image'].height})
                if isinstance(current_vals, int): current_vals = {'top': 0, 'bottom': current_vals}
                
                current_vals['bottom'] = new_bottom
                st.session_state.crop_positions[item['name']] = current_vals
                
                # Update sliders if they exist
                k_bottom = f"slider_bottom_{item['name']}"
                if k_bottom in st.session_state and k_bottom != slider_key_bottom:
                    st.session_state[k_bottom] = new_bottom
                count += 1
            st.success(f"Applied Bottom={new_bottom} to {count} images.")
            st.rerun()

    # Update global crop position state
    st.session_state.crop_positions[current_name] = {'top': new_top, 'bottom': new_bottom}
    
    st.warning("Everything ABOVE Green and BELOW Red will be removed.")
    
    render_navigation("bottom", total_images)

    # --- Final Action ---
    st.divider()
    if st.button("✅ Process & Download All Images", type="primary"):
        output_zip = io.BytesIO()
        
        with zipfile.ZipFile(output_zip, "w") as zf:
            for item in st.session_state.images_data:
                name = item['name']
                img = item['image']
                
                # Get limits
                coords = st.session_state.crop_positions.get(name, {'top':0, 'bottom':img.height})
                if isinstance(coords, int): coords = {'top':0, 'bottom': coords} # Safety
                
                y_top = max(0, int(coords['top']))
                y_bottom = max(1, min(int(coords['bottom']), img.height))
                
                # Validation: Top must be < Bottom
                if y_top >= y_bottom:
                    # Inavlid selection, maybe return original or skip? Let's skip heavy crop.
                    # Or just 0-height? Let's enforce 1px min.
                    y_bottom = y_top + 1
                    if y_bottom > img.height:
                        y_top = img.height - 1
                        y_bottom = img.height
                
                cropped = img.crop((0, y_top, img.width, y_bottom))
                
                img_buffer = io.BytesIO()
                
                # Determine format from original image
                fmt = img.format if img.format else "PNG"
                
                # Save with original format
                try:
                    cropped.save(img_buffer, format=fmt)
                except Exception:
                    # Fallback
                    fmt = "PNG"
                    cropped.save(img_buffer, format=fmt)
                
                # Determine filename in zip
                file_name_in_zip = name
                lower_name = name.lower()
                expected_ext = f".{fmt.lower().replace('jpeg', 'jpg')}"
                if not lower_name.endswith(expected_ext) and "pdf" in lower_name:
                    file_name_in_zip += expected_ext
                
                zf.writestr(file_name_in_zip, img_buffer.getvalue())
                
        st.balloons()
        st.download_button(
            label="Download ZIP",
            data=output_zip.getvalue(),
            file_name="cropped_images.zip",
            mime="application/zip"
        )

if __name__ == "__main__":
    main()
