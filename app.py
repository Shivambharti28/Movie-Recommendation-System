import requests
import streamlit as st
import plotly.express as px
import pandas as pd

# =============================
# CONFIG
# =============================
# app.py
API_BASE = "http://127.0.0.1:8000"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

st.set_page_config(page_title="Movie Recommender & AI Predictor", page_icon="🎬", layout="wide")

# =============================
# STYLES (minimal modern)
# =============================
st.markdown(
    """
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
.small-muted { color:#6b7280; font-size: 0.92rem; }
.movie-title { font-size: 0.9rem; line-height: 1.15rem; height: 2.3rem; overflow: hidden; }
.card { border: 1px solid rgba(0,0,0,0.08); border-radius: 16px; padding: 14px; background: rgba(255,255,255,0.7); }
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# STATE + ROUTING (single-file pages)
# =============================
if "view" not in st.session_state:
    st.session_state.view = "home"  # home | details | predictor
if "selected_tmdb_id" not in st.session_state:
    st.session_state.selected_tmdb_id = None

qp_view = st.query_params.get("view")
qp_id = st.query_params.get("id")
if qp_view in ("home", "details", "predictor"):
    st.session_state.view = qp_view
if qp_id:
    try:
        st.session_state.selected_tmdb_id = int(qp_id)
        st.session_state.view = "details"
    except:
        pass

# Add a navigation button for the Dashboard
st.sidebar.button("📊 ML Dashboard", on_click=lambda: st.session_state.update(view="dashboard"))

# Render the Dashboard View
if st.session_state.view == "dashboard":
    st.title("📊 Movie Analytics & ML Dashboard")
    st.markdown("Explore how our AI groups similar movies together using K-Means Clustering and PCA.")
    
    # Fetch clustering data from FastAPI
    with st.spinner("Loading AI Clustering Data..."):
        try:
            res = requests.get(f"{API_BASE}/api/visualization/clusters")
            if res.status_code == 200:
                cluster_data = res.json().get("clusters", [])
                df_viz = pd.DataFrame(cluster_data)
                
                # Convert 'Cluster' column to string so Plotly treats it as a discrete category (colors)
                df_viz["Cluster"] = df_viz["Cluster"].astype(str)
                
                # Create an interactive Scatter Plot
                fig = px.scatter(
                    df_viz,
                    x="PCA1",
                    y="PCA2",
                    color="Cluster",
                    hover_data=["title"],
                    title="Movie Groupings (K-Means Clustering)",
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                
                # Make the chart look nice
                fig.update_layout(
                    xaxis_title="Principal Component 1",
                    yaxis_title="Principal Component 2",
                    legend_title="Cluster ID"
                )
                
                # Display on the dashboard
                st.plotly_chart(fig, use_container_width=True)
                
                # Optional: Show a quick metric
                st.metric("Total Movies Clustered", len(df_viz))
                
            else:
                st.error("Could not fetch visualization data from the backend.")
        except Exception as e:
            st.error(f"Backend connection error: {e}")






def goto_home():
    st.session_state.view = "home"
    st.query_params["view"] = "home"
    if "id" in st.query_params:
        del st.query_params["id"]
    st.rerun()

def goto_predictor():
    st.session_state.view = "predictor"
    st.query_params["view"] = "predictor"
    if "id" in st.query_params:
        del st.query_params["id"]
    st.rerun()

def goto_details(tmdb_id: int):
    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = int(tmdb_id)
    st.query_params["view"] = "details"
    st.query_params["id"] = str(int(tmdb_id))
    st.rerun()





# =============================
# API HELPERS
# =============================
@st.cache_data(ttl=30)  
def api_get_json(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=25)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {r.text[:300]}"
        return r.json(), None
    except Exception as e:
        return None, f"Request failed: {e}"

def poster_grid(cards, cols=6, key_prefix="grid"):
    if not cards:
        st.info("No movies to show.")
        return

    rows = (len(cards) + cols - 1) // cols
    idx = 0
    for r in range(rows):
        colset = st.columns(cols)
        for c in range(cols):
            if idx >= len(cards):
                break
            m = cards[idx]
            idx += 1

            tmdb_id = m.get("tmdb_id")
            title = m.get("title", "Untitled")
            poster = m.get("poster_url")

            with colset[c]:
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.write("🖼️ No poster")

                if st.button("Open", key=f"{key_prefix}_{r}_{c}_{idx}_{tmdb_id}"):
                    if tmdb_id:
                        goto_details(tmdb_id)

                st.markdown(
                    f"<div class='movie-title'>{title}</div>", unsafe_allow_html=True
                )


def to_cards_from_tfidf_items(tfidf_items):
    cards = []
    for x in tfidf_items or []:
        tmdb = x.get("tmdb") or {}
        if tmdb.get("tmdb_id"):
            cards.append(
                {
                    "tmdb_id": tmdb["tmdb_id"],
                    "title": tmdb.get("title") or x.get("title") or "Untitled",
                    "poster_url": tmdb.get("poster_url"),
                }
            )
    return cards


def parse_tmdb_search_to_cards(data, keyword: str, limit: int = 24):
    keyword_l = keyword.strip().lower()

    if isinstance(data, dict) and "results" in data:
        raw = data.get("results") or []
        raw_items = []
        for m in raw:
            title = (m.get("title") or "").strip()
            tmdb_id = m.get("id")
            poster_path = m.get("poster_path")
            if not title or not tmdb_id:
                continue
            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": f"{TMDB_IMG}{poster_path}" if poster_path else None,
                    "release_date": m.get("release_date", ""),
                }
            )
    elif isinstance(data, list):
        raw_items = []
        for m in data:
            tmdb_id = m.get("tmdb_id") or m.get("id")
            title = (m.get("title") or "").strip()
            poster_url = m.get("poster_url")
            if not title or not tmdb_id:
                continue
            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": poster_url,
                    "release_date": m.get("release_date", ""),
                }
            )
    else:
        return [], []

    matched = [x for x in raw_items if keyword_l in x["title"].lower()]
    final_list = matched if matched else raw_items

    suggestions = []
    for x in final_list[:10]:
        year = (x.get("release_date") or "")[:4]
        label = f"{x['title']} ({year})" if year else x["title"]
        suggestions.append((label, x["tmdb_id"]))

    cards = [
        {"tmdb_id": x["tmdb_id"], "title": x["title"], "poster_url": x["poster_url"]}
        for x in final_list[:limit]
    ]
    return suggestions, cards


# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.markdown("## 🎬 Navigation")
    if st.button("🏠 Home Recommender"):
        goto_home()
    
    if st.button("📈 Blockbuster Predictor"):
        goto_predictor()

    st.markdown("---")
    st.markdown("### 🏠 Home Feed Settings")
    home_category = st.selectbox(
        "Category",
        ["trending", "popular", "top_rated", "now_playing", "upcoming"],
        index=0,
    )
    grid_cols = st.slider("Grid columns", 4, 8, 6)

# =============================
# HEADER
# =============================
if st.session_state.view != "predictor":
    st.title("🎬 Movie Recommender")
    st.markdown(
        "<div class='small-muted'>Type keyword → dropdown suggestions + matching results → open → details + recommendations</div>",
        unsafe_allow_html=True,
    )
    st.divider()

# ==========================================================
# VIEW: PREDICTOR (New Machine Learning Feature)
# ==========================================================
if st.session_state.view == "predictor":
    st.title("📈 The Blockbuster Predictor")
    st.markdown("Will your hypothetical movie be a commercial hit? Enter the production details below to find out using our **Random Forest Classification Model**.")
    st.divider()

    with st.form("hit_predictor_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            budget_input = st.number_input("Production Budget ($)", min_value=1000, value=50000000, step=1000000)
        with col2:
            runtime_input = st.number_input("Runtime (minutes)", min_value=10, value=120, step=5)
        with col3:
            popularity_input = st.number_input("Expected TMDB Popularity", min_value=0.0, value=20.0, step=1.0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Box Office Fate 🎬", use_container_width=True)

    if submitted:
        with st.spinner("Consulting the Machine Learning Model..."):
            try:
                payload = {
                    "budget": float(budget_input), 
                    "runtime": float(runtime_input), 
                    "popularity": float(popularity_input)
                }
                # Call the new FastAPI Endpoint
                r = requests.post(f"{API_BASE}/predict/hit", json=payload, timeout=10)
                
                if r.status_code == 200:
                    res = r.json()
                    is_hit = res.get("is_hit")
                    prob = res.get("probability", 0.0)
                    msg = res.get("message", "")

                    st.markdown("### Prediction Results")
                    c1, c2 = st.columns(2)
                    with c1:
                        if is_hit:
                            st.success(f"### 🎉 {msg}")
                        else:
                            st.error(f"### 📉 {msg}")
                    with c2:
                        st.metric("Model Confidence (Probability)", f"{prob * 100:.1f}%")
                        
                else:
                    st.error(f"Error from API (Are your ML pickles loaded?): {r.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")

# ==========================================================
# VIEW: HOME
# ==========================================================
elif st.session_state.view == "home":
    typed = st.text_input(
        "Search by movie title (keyword)", placeholder="Type: avenger, batman, love..."
    )

    st.divider()

    # SEARCH MODE
    if typed.strip():
        st.markdown(f"### 🧠 AI Keyword Matches for: *'{typed}'*")
        
        # Hit your new custom NLP endpoint instead of TMDB!
        nlp_cards, err = api_get_json("/nlp/keyword_search", params={"query": typed.strip(), "limit": 12})
        
        if err:
            st.error(f"NLP Engine Error: {err}")
        elif not nlp_cards:
            st.warning("No semantic matches found. Try different keywords!")
        else:
            # We bypass the 'suggestions' dropdown and just show the AI matches instantly
            poster_grid(nlp_cards, cols=grid_cols, key_prefix="nlp_results")
            
        st.stop()

    # HOME FEED MODE
    st.markdown(f"### 🏠 Home — {home_category.replace('_',' ').title()}")

    home_cards, err = api_get_json(
        "/home", params={"category": home_category, "limit": 24}
    )
    if err or not home_cards:
        st.error(f"Home feed failed: {err or 'Unknown error'}")
        st.stop()

    poster_grid(home_cards, cols=grid_cols, key_prefix="home_feed")

# ==========================================================
# VIEW: DETAILS
# ==========================================================
elif st.session_state.view == "details":
    tmdb_id = st.session_state.selected_tmdb_id
    if not tmdb_id:
        st.warning("No movie selected.")
        if st.button("← Back to Home"):
            goto_home()
        st.stop()

    a, b = st.columns([3, 1])
    with a:
        st.markdown("### 📄 Movie Details")
    with b:
        if st.button("← Back to Home"):
            goto_home()

    data, err = api_get_json(f"/movie/id/{tmdb_id}")
    if err or not data:
        st.error(f"Could not load details: {err or 'Unknown error'}")
        st.stop()

    left, right = st.columns([1, 2.4], gap="large")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if data.get("poster_url"):
            st.image(data["poster_url"], use_container_width=True)
        else:
            st.write("🖼️ No poster")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"## {data.get('title','')}")
        release = data.get("release_date") or "-"
        genres = ", ".join([g["name"] for g in data.get("genres", [])]) or "-"
        st.markdown(
            f"<div class='small-muted'>Release: {release}</div>", unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='small-muted'>Genres: {genres}</div>", unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown("### Overview")
        st.write(data.get("overview") or "No overview available.")
        st.markdown("</div>", unsafe_allow_html=True)

    if data.get("backdrop_url"):
        st.markdown("#### Backdrop")
        st.image(data["backdrop_url"], use_container_width=True)

    st.divider()
    st.markdown("### ✅ Recommendations")

    title = (data.get("title") or "").strip()
    if title:
        bundle, err2 = api_get_json(
            "/movie/search",
            params={"query": title, "tfidf_top_n": 12, "genre_limit": 12},
        )

        if not err2 and bundle:
            st.markdown("#### 🔎 Similar Movies (TF-IDF)")
            poster_grid(
                to_cards_from_tfidf_items(bundle.get("tfidf_recommendations")),
                cols=grid_cols,
                key_prefix="details_tfidf",
            )

            st.markdown("#### 🎭 More Like This (Genre)")
            poster_grid(
                bundle.get("genre_recommendations", []),
                cols=grid_cols,
                key_prefix="details_genre",
            )
        else:
            st.info("Showing Genre recommendations (fallback).")
            genre_only, err3 = api_get_json(
                "/recommend/genre", params={"tmdb_id": tmdb_id, "limit": 18}
            )
            if not err3 and genre_only:
                poster_grid(
                    genre_only, cols=grid_cols, key_prefix="details_genre_fallback"
                )
            else:
                st.warning("No recommendations available right now.")
    else:
        st.warning("No title available to compute recommendations.")