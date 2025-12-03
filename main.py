import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import json
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
import re
import pytesseract

def detect_red_points(image_array):
    """
    Détecte automatiquement les points rouges sur l'image
    """
    if len(image_array.shape) == 3:
        # Convertir RGB en HSV
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # Définir la plage de couleur rouge
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Créer les masques
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Nettoyer le masque
        kernel = np.ones((3,3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # Détecter les contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extraire les centres
        red_points = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 1000:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    red_points.append((cx, cy))
        
        red_points.sort(key=lambda p: (p[1], p[0]))
        return red_points
    
    return []

def main():
    st.title("Analyseur de Points Anatomiques - Radiologie Dentaire")
    st.markdown("---")
    
    # Instructions simplifiées
    with st.expander("Instructions d'utilisation", expanded=False):
        st.markdown("""
        **Workflow simplifié :**
        1. **Uploadez** votre radiographie avec points rouges
        2. **Cliquez** sur "Détecter les points rouges" 
        3. **Sélectionnez** votre point de référence S dans la liste
        4. **Renommez** les points selon vos conventions
        5. **Exportez** les résultats en Excel
        
        **Facteur de conversion :** 10 mm = 4545 pixels
        """)

    uploaded_file = st.file_uploader(
        "Choisir une radiographie", 
        type=['png', 'jpg', 'jpeg', 'bmp']
    )
    
    pixel_to_mm = 1 / 4.545
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Initialiser session state
        if 'detected_points' not in st.session_state:
            st.session_state.detected_points = []
        if 'point_names' not in st.session_state:
            st.session_state.point_names = []
        if 'reference_point_index' not in st.session_state:
            st.session_state.reference_point_index = None
        if 'hovered_point_index' not in st.session_state:
            st.session_state.hovered_point_index = None
        
        # Boutons de contrôle
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("Détecter les points rouges", type="primary"):
                with st.spinner("Détection des points rouges en cours..."):
                    detected_points = detect_red_points(img_array)
                    
                    if detected_points:
                        st.session_state.detected_points = detected_points
                        st.session_state.point_names = [f'P{i+1}' for i in range(len(detected_points))]
                        st.session_state.reference_point_index = None
                        st.success(f"{len(detected_points)} points rouges détectés !")
                        st.rerun()
                    else:
                        st.error("Aucun point rouge détecté. Vérifiez votre image.")
        
        with col2:
            if st.button("Réinitialiser", type="secondary"):
                st.session_state.detected_points = []
                st.session_state.point_names = []
                st.session_state.reference_point_index = None
                st.session_state.hovered_point_index = None
                st.rerun()
        
        with col3:
            if st.session_state.detected_points:
                st.info(f"{len(st.session_state.detected_points)} points détectés")
            else:
                st.info("Cliquez sur 'Détecter les points rouges' pour commencer")
        
        # Affichage principal
        if st.session_state.detected_points:
            col_image, col_table = st.columns([0.8, 1.2])
            
            with col_image:
                # Créer l'image avec les points annotés
                display_image = image.copy()
                draw = ImageDraw.Draw(display_image)
                
                # Dessiner tous les points détectés
                for i, (point_x, point_y) in enumerate(st.session_state.detected_points):
                    if i == st.session_state.reference_point_index:
                        color = 'green'
                        outline_color = 'darkgreen'
                        circle_radius = 15
                        outline_width = 4
                        point_name = 'S'
                    elif st.session_state.hovered_point_index == i:
                        color = 'cyan'
                        outline_color = 'blue'
                        circle_radius = 12
                        outline_width = 3
                        point_name = st.session_state.point_names[i] if i < len(st.session_state.point_names) else f'P{i+1}'
                    else:
                        color = 'red'
                        outline_color = 'darkred'
                        circle_radius = 8
                        outline_width = 2
                        point_name = st.session_state.point_names[i] if i < len(st.session_state.point_names) else f'P{i+1}'
                    
                    # Dessiner le point
                    draw.ellipse([point_x-circle_radius, point_y-circle_radius, 
                                point_x+circle_radius, point_y+circle_radius], 
                               fill=color, outline=outline_color, width=outline_width)
                    
                    # Texte avec contour blanc pour visibilité
                    text_x, text_y = point_x + 25, point_y - 8
                    for offset in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                        draw.text((text_x + offset[0], text_y + offset[1]), point_name, fill='white')
                    draw.text((text_x, text_y), point_name, fill=color)
                
                # Règle graduée
                ruler_length_mm = 50
                ruler_length_px = int(ruler_length_mm / pixel_to_mm)
                ruler_start_x = 50
                ruler_start_y = img_array.shape[0] - 50
                ruler_end_x = ruler_start_x + ruler_length_px
                
                draw.line([(ruler_start_x, ruler_start_y), (ruler_end_x, ruler_start_y)], fill='white', width=4)
                draw.line([(ruler_start_x, ruler_start_y), (ruler_end_x, ruler_start_y)], fill='black', width=2)
                
                for mm in range(0, ruler_length_mm + 1, 10):
                    x_pos = ruler_start_x + int(mm / pixel_to_mm)
                    draw.line([(x_pos, ruler_start_y - 6), (x_pos, ruler_start_y + 6)], fill='white', width=3)
                    draw.line([(x_pos, ruler_start_y - 4), (x_pos, ruler_start_y + 4)], fill='black', width=1)
                    if mm % 20 == 0:
                        draw.text((x_pos - 5, ruler_start_y - 20), f'{mm}', fill='white')
                        draw.text((x_pos - 4, ruler_start_y - 19), f'{mm}', fill='black')
                
                draw.text((ruler_end_x + 5, ruler_start_y - 10), 'mm', fill='white')
                draw.text((ruler_end_x + 6, ruler_start_y - 9), 'mm', fill='black')
                
                # Affichage de l'image avec qualité préservée
                st.image(display_image, use_container_width=True)
            
            with col_table:
                # Sélection du point de référence S
                if st.session_state.reference_point_index is None:
                    st.markdown("**Sélectionner le point de référence S:**")
                    ref_options = [(i, f"P{i+1}: ({st.session_state.detected_points[i][0]}, {st.session_state.detected_points[i][1]})") 
                                 for i in range(len(st.session_state.detected_points))]
                    
                    selected_ref = st.selectbox(
                        "Choisir le point S:",
                        options=[opt[0] for opt in ref_options],
                        format_func=lambda x: ref_options[x][1],
                        key="ref_selection"
                    )

                    if selected_ref == "Aucun":
                        st.session_state.hovered_point_index = None
                    else:
                        point_index = int(selected_ref)
                        if st.session_state.hovered_point_index != point_index:
                            st.session_state.hovered_point_index = point_index
                            st.rerun()

                    
                    if st.button("Confirmer le point S", type="primary"):
                        st.session_state.reference_point_index = selected_ref
                        st.session_state.point_names[selected_ref] = 'S'
                        st.success(f"Point S défini")
                        st.rerun()
                
                else:
                    # Tableau avec validation
                    ref_x, ref_y = st.session_state.detected_points[st.session_state.reference_point_index]
                    VALID_NAMES = ['S', 'Po', 'Ba', 'Ar', 'Go', 'Me', "Me'", 'Gn', 'Pog', "Pog'", 'B', 'A', 'UL', 'LL', 'Co', 'Prn', 'Na', "Na'", 'Or', 'PNS', 'ANS']
                    
                    # Sélection pour surbrillance (format compact)
                    radio_options = ["Aucun"] + [f"P{i+1}" for i in range(len(st.session_state.detected_points))]
                    
                    current_selection = "Aucun"
                    if st.session_state.hovered_point_index is not None:
                        current_selection = f"P{st.session_state.hovered_point_index + 1}"
                    
                    selected_option = st.radio(
                        "Surbrillance:",
                        options=radio_options,
                        index=radio_options.index(current_selection),
                        horizontal=True,
                        key="highlight_radio"
                    )
                    
                    # Mettre à jour la surbrillance
                    if selected_option == "Aucun":
                        st.session_state.hovered_point_index = None
                    else:
                        point_index = int(selected_option[1:]) - 1
                        if st.session_state.hovered_point_index != point_index:
                            st.session_state.hovered_point_index = point_index
                            st.rerun()
                    
                    # Tableau des données
                    table_data = []
                    for i, (point_x, point_y) in enumerate(st.session_state.detected_points):
                        rel_x_px = point_x - ref_x
                        rel_y_px = ref_y - point_y ### TODO
                        rel_x_mm = rel_x_px * pixel_to_mm
                        rel_y_mm = rel_y_px * pixel_to_mm
                        
                        current_name = st.session_state.point_names[i] if i < len(st.session_state.point_names) else f'P{i+1}'
                        
                        table_data.append({
                            'Nom': current_name,
                            'X abs': point_x,
                            'Y abs': point_y,
                            'X rel': rel_x_px,
                            'Y rel': rel_y_px,
                            'X mm': round(rel_x_mm, 2),
                            'Y mm': round(rel_y_mm, 2)
                        })
                    
                    df_table = pd.DataFrame(table_data)
                    
                    edited_df = st.data_editor(
                        df_table,
                        column_config={
                            "Nom": st.column_config.TextColumn("Nom", width="small"),
                            #"X abs": st.column_config.NumberColumn("X abs", disabled=True, width="small"),
                            #"Y abs": st.column_config.NumberColumn("Y abs", disabled=True, width="small"),
                            "X rel": st.column_config.NumberColumn("X rel", disabled=True, width="small"),
                            "Y rel": st.column_config.NumberColumn("Y rel", disabled=True, width="small"),
                            "X mm": st.column_config.NumberColumn("X mm", disabled=True, width="small"),
                            "Y mm": st.column_config.NumberColumn("Y mm", disabled=True, width="small"),
                        },
                        disabled=["X abs", "Y abs", "X rel", "Y rel", "X mm", "Y mm"],
                        use_container_width=True,
                        height=500,
                        key="points_table",
                        hide_index=True
                    )
                    
                    # Validation et mise à jour
                    validation_errors = []
                    names_used = []
                    
                    for i, row in edited_df.iterrows():
                        new_name = row['Nom'].strip()
                        
                        if i < len(st.session_state.point_names) and st.session_state.point_names[i] != new_name:
                            st.session_state.point_names[i] = new_name
                            st.rerun()
                        
                        if new_name and new_name not in VALID_NAMES:
                            validation_errors.append(f"'{new_name}' invalide (P{i+1})")
                        
                        if new_name and new_name in names_used:
                            validation_errors.append(f"'{new_name}' en double")
                        
                        if new_name:
                            names_used.append(new_name)
                    
                    if validation_errors:
                        st.error("Erreurs:")
                        for error in validation_errors:
                            st.write(f"• {error}")
                        st.info(f"Noms valides: {', '.join(VALID_NAMES)}")
                    else:
                        st.success("Tous les noms sont valides")

            # Export
            if st.session_state.reference_point_index is not None:
                st.markdown("### Export des Résultats")
                
                ref_x, ref_y = st.session_state.detected_points[st.session_state.reference_point_index]
                relative_coords = []
                
                for i, (point_x, point_y) in enumerate(st.session_state.detected_points):
                    rel_x_px = point_x - ref_x
                    rel_y_px = point_y - ref_y
                    rel_x_mm = rel_x_px * pixel_to_mm
                    rel_y_mm = rel_y_px * pixel_to_mm
                    
                    point_name = st.session_state.point_names[i] if i < len(st.session_state.point_names) else f'P{i+1}'
                    
                    relative_coords.append({
                        'Point': point_name,
                        'X_absolu_px': point_x,
                        'Y_absolu_px': point_y,
                        'X_relatif_px': rel_x_px,
                        'Y_relatif_px': rel_y_px,
                        'X_relatif_mm': round(rel_x_mm, 2),
                        'Y_relatif_mm': round(rel_y_mm, 2)
                    })
                
                df_results = pd.DataFrame(relative_coords)
                st.dataframe(df_results, use_container_width=True)
                
                # Export Excel
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_results.to_excel(writer, sheet_name='Points_Anatomiques', index=False)
                    
                    info_data = {
                        'Information': ['Nom image', 'Point S (X,Y)', 'Facteur (mm/px)', 'Nb points', 'Date'],
                        'Valeur': [
                            uploaded_file.name,
                            f"({ref_x}, {ref_y})",
                            pixel_to_mm,
                            len(st.session_state.detected_points),
                            pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        ]
                    }
                    df_info = pd.DataFrame(info_data)
                    df_info.to_excel(writer, sheet_name='Informations', index=False)
                
                st.download_button(
                    label="Télécharger Excel",
                    data=buffer.getvalue(),
                    file_name=f"analyse_points_{uploaded_file.name.split('.')[0]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
    
    st.markdown(
        """
        <hr>
        <div style="text-align: center; font-size: 0.9rem; color: #666;">
            Application développée par <strong>Dr. Douae Ahmadoun</strong>  
            pour <strong>Dr. Afaf Merouani Idrissi</strong> dans le cadre de ses travaux sur les images radiologiques orthodontiques.
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    st.set_page_config(
        page_title="Analyseur Radiologie Dentaire",
        page_icon="🦷",
        layout="wide"
    )
    main()
