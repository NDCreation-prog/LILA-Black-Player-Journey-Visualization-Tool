import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import pyarrow.parquet as pq
from sklearn.cluster import KMeans

# --- 1. CONFIGURATION ---
MAP_CONFIG = {
    'AmbroseValley': {'scale': 900, 'ox': -370, 'oz': -473, 'img': 'minimaps/AmbroseValley_Minimap.png'},
    'GrandRift': {'scale': 581, 'ox': -290, 'oz': -290, 'img': 'minimaps/GrandRift_Minimap.png'},
    'Lockdown': {'scale': 1000, 'ox': -500, 'oz': -500, 'img': 'minimaps/Lockdown_Minimap.jpg'}
}

EVENT_STYLES = {
    'Kill': {'color': 'red', 'symbol': 'x', 'size': 12},
    'Death': {'color': 'black', 'symbol': 'skull', 'size': 12},
    'Loot': {'color': 'gold', 'symbol': 'diamond', 'size': 10},
    'Storm_Death': {'color': 'purple', 'symbol': 'cloud', 'size': 12}
}


st.set_page_config(layout="wide", page_title="LILA BLACK AI Designer")

# --- 2. DATA PREPROCESSING ---
def load_day(folder_path):
    frames = []
    for f in os.listdir(folder_path):
        filepath = os.path.join(folder_path, f)
        try:
            table = pq.read_table(filepath)
            df = table.to_pandas()
            if 'event' in df.columns:
                df['event'] = df['event'].apply(
                    lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
                )
            frames.append(df)
        except Exception as e:
            print("Skipping:", filepath)
    return pd.concat(frames, ignore_index=True)

@st.cache_data
def load_and_preprocess_raw_data(base_folder="New_player_data"):
    if not os.path.exists(base_folder):
        return None
    
    all_days_data = []
    for day_folder in os.listdir(base_folder):
        day_path = os.path.join(base_folder, day_folder)
        if os.path.isdir(day_path):
            day_df = load_day(day_path)
            day_df["day"] = day_folder
            all_days_data.append(day_df)
    
    if not all_days_data:
        return None
        
    df_all = pd.concat(all_days_data, ignore_index=True)
    return df_all

# --- 3. DATA & AI ENGINE ---
@st.cache_data
def load_and_cluster():
    # Attempt to load processed CSV first, otherwise process raw data
    file_path = "all_player_data_Org.csv"
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = load_and_preprocess_raw_data()
        if df is not None:
            df.to_csv(file_path, index=False)
    
    if df is None:
        return None
        
    df['ts'] = pd.to_datetime(df['ts'])
    
    # ts_numeric is used for the playback timeline
    df['ts_numeric'] = df.groupby('match_id')['ts'].transform(lambda x: (x - x.min()).dt.total_seconds() * 1000)
    df['event'] = df['event'].astype(str)

    # Distinguish Bots vs Humans
    if 'is_bot' not in df.columns:
        df['is_bot'] = df['user_id'].astype(str).str.contains('bot', case=False)

    # AI FEATURE ENGINEERING: Clustering Playstyles
    player_behavior = df.groupby('user_id').agg({
        'x': 'std', 
        'z': 'std', 
        'event': lambda x: (x == 'Kill').sum()
    }).fillna(0)
    
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    player_behavior['cluster'] = kmeans.fit_predict(player_behavior)
    
    persona_map = player_behavior['cluster'].to_dict()
    df['persona_id'] = df['user_id'].map(persona_map)
    
    persona_names = {0: "Tactical (Low Movement)", 1: "Aggressive (High Kills)", 2: "Explorer (High Movement)"}
    df['persona_name'] = df['persona_id'].map(persona_names).fillna("Unknown")
    
    return df

def to_pixel(x, z, map_id):
    cfg = MAP_CONFIG[map_id]
    u = (x - cfg['ox']) / cfg['scale']
    v = (z - cfg['oz']) / cfg['scale']
    return u * 1024, (1 - v) * 1024

# --- 4. UI LAYOUT ---
df = load_and_cluster()

if df is not None:
    st.sidebar.title("AI Analytics Dashboard")
    
    selected_map = st.sidebar.selectbox("Select Map", list(MAP_CONFIG.keys()))
    
    available_days = sorted(df['day'].unique())
    selected_day = st.sidebar.selectbox("Filter by Date", available_days)
    
    map_df = df[(df['map_id'] == selected_map) & (df['day'] == selected_day)].copy()
    
    if map_df.empty:
        st.warning(f"No data found for {selected_day} on {selected_map}.")
    else:
        map_df['px_x'], map_df['px_y'] = to_pixel(map_df['x'], map_df['z'], selected_map)
        match_ids = map_df['match_id'].unique()
        selected_match = st.sidebar.selectbox("Select Match ID", match_ids)
        
        st.sidebar.divider()
        show_players = st.sidebar.multiselect("Show Player Types", ["Human", "Bot"], default=["Human", "Bot"])
        player_filter = []
        if "Human" in show_players: player_filter.append(False)
        if "Bot" in show_players: player_filter.append(True)

        analysis_mode = st.sidebar.radio("Analysis Mode", ["AI Playstyle Journey", "Heatmap Analysis"])
        all_personas = df['persona_name'].unique()
        selected_personas = st.sidebar.multiselect("Filter Personas", all_personas, default=list(all_personas))

        match_df = map_df[(map_df['match_id'] == selected_match) & (map_df['is_bot'].isin(player_filter))]
        
        st.title(f"Level Intelligence: {selected_map}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Active Personas", len(match_df['persona_name'].unique()))
        m2.metric("Total Match Events", len(match_df))
        m3.metric("Duration", f"{round(match_df['ts_numeric'].max()/60000, 1) if not match_df.empty else 0}m")
        m4.metric("Bots in Match", match_df['is_bot'].sum())

        if analysis_mode == "AI Playstyle Journey":
            max_ms = int(match_df['ts_numeric'].max()) if not match_df.empty else 0
            
            if max_ms > 0:
                current_time = st.slider("Scrub Match Timeline (ms)", 0, max_ms, max_ms)
            else:
                st.info("Showing all match points.")
                current_time = 0
            
            display_df = match_df[(match_df['ts_numeric'] <= current_time) & (match_df['persona_name'].isin(selected_personas))]
            
            fig = go.Figure()
            
            img_path = MAP_CONFIG[selected_map]['img']
            if os.path.exists(img_path):
                img = Image.open(img_path)
                fig.add_layout_image(dict(source=img, x=0, y=0, sizex=1024, sizey=1024, xref="x", yref="y", sizing="stretch", layer="below"))

            persona_colors = {"Tactical (Low Movement)": "#FFFF00", "Aggressive (High Kills)": "#FF0000", "Explorer (High Movement)": "#00FF00"}
            
            for uid in display_df['user_id'].unique():
                p_df = display_df[display_df['user_id'] == uid]
                p_name = p_df['persona_name'].iloc[0]
                is_bot = p_df['is_bot'].iloc[0]
                
                fig.add_trace(go.Scatter(
                    x=p_df['px_x'], y=p_df['px_y'],
                    mode='lines',
                    line=dict(color=persona_colors.get(p_name, "white"), width=2, dash='dot' if is_bot else 'solid'),
                    name=f"{'Bot' if is_bot else 'Player'}: {uid}",
                    legendgroup=p_name,
                    showlegend=False
                ))

                events_in_trace = p_df[p_df['event'].isin(EVENT_STYLES.keys())]
                for event_type, style in EVENT_STYLES.items():
                    e_df = events_in_trace[events_in_trace['event'] == event_type]
                    if not e_df.empty:
                        fig.add_trace(go.Scatter(
                            x=e_df['px_x'], y=e_df['px_y'],
                            mode='markers',
                            marker=dict(color=style['color'], symbol=style['symbol'], size=style['size']),
                            name=event_type,
                            showlegend=True if uid == display_df['user_id'].unique()[0] else False
                        ))

            fig.update_xaxes(range=[0, 1024], visible=False)
            fig.update_yaxes(range=[1024, 0], visible=False)
            fig.update_layout(width=900, height=850, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.subheader("Spatial Density Analysis")
            heat_layer = st.selectbox("Select Heatmap Layer", ["Kills", "Deaths", "High Traffic (Movement)"])
            
            if heat_layer == "Kills":
                h_df = map_df[map_df['event'].str.contains('Kill', case=False)]
                colors = "Hot"
            elif heat_layer == "Deaths":
                h_df = map_df[map_df['event'].str.contains('Death', case=False)]
                colors = "Redor"
            else:
                h_df = map_df 
                colors = "Viridis"
            
            if not h_df.empty:
                fig = px.density_heatmap(
                    h_df, x="px_x", y="px_y", 
                    nbinsx=60, nbinsy=60, 
                    color_continuous_scale=colors,
                    range_x=[0, 1024], range_y=[0, 1024]
                )
                fig.update_traces(opacity=0.6)

                img_path = MAP_CONFIG[selected_map]['img']
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    fig.add_layout_image(dict(source=img, x=0, y=0, sizex=1024, sizey=1024, xref="x", yref="y", sizing="stretch", layer="below"))

                fig.update_xaxes(range=[0, 1024], visible=False)
                fig.update_yaxes(range=[1024, 0], visible=False)
                fig.update_layout(width=900, height=850)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected heatmap layer.")

else:
    st.error("Data source not found. Please ensure 'all_player_data.csv' or the 'New_player_data' folder is available.")