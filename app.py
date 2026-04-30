import requests
import streamlit as st
import plotly.express as px
import pandas as pd

# =============================
# CONFIG
# =============================
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
# STATE + ROUTING
# =============================
if "view" not in st.session_state:
    st.session_state.view = "home"  # home | details | predictor | dashboard
if "selected_tmdb_id" not in st.session_state:
    st.session_state.selected_tmdb_id = None

qp_view = st.query_params.get("view")
qp_id = st.query_params.get("id")
if qp_view in ("home", "details", "predictor", "dashboard"):
    st.session_state.view = qp_view
if qp_id:
    try:
        st.session_state.selected_tmdb_id = int(qp_id)
        st.session_state.view = "details"
    except:
        pass


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

def goto_dashboard():
    st.session_state.view = "dashboard"
    st.query_params["view"] = "dashboard"
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
        st.info("No movies to show based on your current filters.")
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
    if st.button("🏠 Home Recommender", use_container_width=True):
        goto_home()
    
    if st.button("📈 Revenue Predictor", use_container_width=True):
        goto_predictor()
        
    if st.button("📊 Dashboard", use_container_width=True):
        goto_dashboard()

    if st.session_state.view == "home":
        st.markdown("---")
        st.markdown("### 🏠 Home Feed Settings")
        
        # Mapped to distinct, useful endpoints
        category_map = {
            "Now Playing (Theaters)": "now_playing",
            "All-Time Top Rated": "top_rated",
            "Upcoming Releases": "upcoming"
        }
        
        selected_category_label = st.selectbox(
            "Collection",
            list(category_map.keys()),
            index=0,
        )
        home_category = category_map[selected_category_label]
        
        # Replaced grid columns with a functional filter
        min_rating = st.slider("Minimum Rating (0-10)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)

# ==========================================================
# VIEW: DASHBOARD (Exploratory Data Analysis)
# ==========================================================
if st.session_state.view == "dashboard":
    st.title("📊 Interactive Dashboard")
    st.markdown("Explore the cinematic dataset driving our machine learning models. Use the filters below to slice the data and watch the charts update dynamically.")
    st.divider()
    
    with st.spinner("Loading Dashboard Data..."):
        try:
            res = requests.get(f"{API_BASE}/api/visualization/eda")
            if res.status_code == 200:
                eda_data = res.json().get("data", [])
                df_viz = pd.DataFrame(eda_data)
                
                # Map Revenue Tiers for colors
                tier_mapping = {0: "Low Revenue", 1: "Medium Revenue", 2: "High Revenue"}
                df_viz["Revenue Tier"] = df_viz["revenue_tier"].map(tier_mapping)
                color_map = {"Low Revenue": "#EF553B", "Medium Revenue": "#FECB52", "High Revenue": "#00CC96"}

                # --- INTERACTIVE FILTERS ---
                st.markdown("### 🎛️ Data Filters")
                filt_col1, filt_col2 = st.columns(2)
                with filt_col1:
                    selected_tiers = st.multiselect(
                        "Filter by AI Revenue Tier", 
                        options=["Low Revenue", "Medium Revenue", "High Revenue"], 
                        default=["Low Revenue", "Medium Revenue", "High Revenue"]
                    )
                with filt_col2:
                    max_budget = st.slider(
                        "Maximum Production Budget ($)", 
                        min_value=int(df_viz['budget'].min()), 
                        max_value=int(df_viz['budget'].max()), 
                        value=int(df_viz['budget'].max()), 
                        step=10000000
                    )
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Apply Filters
                filtered_df = df_viz[
                    (df_viz["Revenue Tier"].isin(selected_tiers)) & 
                    (df_viz["budget"] <= max_budget)
                ]

                # Prevent crash if filtering removes all data
                if filtered_df.empty:
                    st.warning("⚠️ No movies match your current filters. Adjust your sliders or selections.")
                else:
                    # --- ROW 1: TOP LEVEL METRICS ---
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Movies in View", f"{len(filtered_df):,}")
                    col2.metric("Average Budget", f"${filtered_df['budget'].mean():,.0f}")
                    col3.metric("Average Box Office", f"${filtered_df['revenue'].mean():,.0f}")
                    col4.metric("Average Rating", f"{filtered_df['vote_average'].mean():.1f}/10")
                    
                    st.markdown("<br>", unsafe_allow_html=True)

                    # --- ROW 2: MAIN SCATTER PLOT ---
                    st.markdown("### Budget vs. Revenue")
                    fig_scatter = px.scatter(
                        filtered_df,
                        x="budget",
                        y="revenue",
                        color="Revenue Tier",
                        hover_data=["title", "popularity", "vote_average"],
                        labels={"budget": "Production Budget ($)", "revenue": "Box Office Revenue ($)"},
                        color_discrete_map=color_map,
                        log_x=True, 
                        log_y=True,
                        height=500
                    )
                    fig_scatter.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_scatter, use_container_width=True)

                    st.markdown("<hr>", unsafe_allow_html=True)

                    # --- ROW 3: PIE CHART & HISTOGRAM ---
                    col_left, col_right = st.columns(2)
                    
                    with col_left:
                        st.markdown("### Revenue Tier Distribution")
                        tier_counts = filtered_df["Revenue Tier"].value_counts().reset_index()
                        tier_counts.columns = ["Revenue Tier", "Count"]
                        fig_pie = px.pie(
                            tier_counts, 
                            names="Revenue Tier", 
                            values="Count", 
                            color="Revenue Tier",
                            color_discrete_map=color_map,
                            hole=0.4
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col_right:
                        st.markdown("### Audience Rating Distribution")
                        fig_hist = px.histogram(
                            filtered_df, 
                            x="vote_average", 
                            nbins=30, 
                            color_discrete_sequence=["#636EFA"],
                            labels={"vote_average": "TMDB Vote Average (0-10)"}
                        )
                        fig_hist.update_layout(yaxis_title="Number of Movies")
                        st.plotly_chart(fig_hist, use_container_width=True)

                    st.markdown("<hr>", unsafe_allow_html=True)

                    # --- ROW 4: TOP 10 LEADERBOARD ---
                    st.markdown("### Top 10 Most Popular Movies (in current view)")
                    top_10_pop = filtered_df.nlargest(10, 'popularity').sort_values('popularity', ascending=True)
                    fig_bar = px.bar(
                        top_10_pop, 
                        x='popularity', 
                        y='title', 
                        orientation='h',
                        color='popularity',
                        color_continuous_scale='Viridis',
                        labels={'popularity': 'TMDB Popularity Score', 'title': 'Movie Title'}
                    )
                    fig_bar.update_layout(height=450, showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)

            else:
                st.error("Could not fetch visualization data from the backend.")
        except Exception as e:
            st.error(f"Backend connection error: {e}")

# ==========================================================
# VIEW: PREDICTOR (Logistic Regression Feature)
# ==========================================================
elif st.session_state.view == "predictor":
    st.title("📈 The Revenue Tier Predictor")
    st.markdown("Will your hypothetical movie be a hit? Enter the production details below to find out using our **Logistic Regression Model**.")
    st.divider()

    with st.form("revenue_predictor_form"):
        st.markdown("### Movie Attributes")
        col1, col2 = st.columns(2)
        with col1:
            budget_input = st.number_input("Production Budget ($)", min_value=1000, value=50000000, step=1000000)
            runtime_input = st.number_input("Runtime (minutes)", min_value=10, value=120, step=5)
            popularity_input = st.number_input("Expected TMDB Popularity", min_value=0.0, value=20.0, step=1.0)
        with col2:
            vote_average_input = st.number_input("Expected Vote Average (0-10)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
            vote_count_input = st.number_input("Expected Vote Count", min_value=0, value=1500, step=100)
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Box Office Fate 🎬", use_container_width=True)

    if submitted:
        with st.spinner("Consulting the Machine Learning Model..."):
            try:
                payload = {
                    "budget": float(budget_input), 
                    "runtime": float(runtime_input), 
                    "popularity": float(popularity_input),
                    "vote_average": float(vote_average_input),
                    "vote_count": float(vote_count_input)
                }
                r = requests.post(f"{API_BASE}/predict/revenue", json=payload, timeout=10)
                
                if r.status_code == 200:
                    res = r.json()
                    tier_name = res.get("tier_name")
                    prob = res.get("probability", 0.0)
                    msg = res.get("message", "")

                    st.markdown("### Prediction Results")
                    c1, c2 = st.columns(2)
                    with c1:
                        if tier_name == "High Revenue":
                            st.success(f"### 🎉 {msg}")
                        elif tier_name == "Medium Revenue":
                            st.warning(f"### 📊 {msg}")
                        else:
                            st.error(f"### 📉 {msg}")
                    with c2:
                        st.metric("Model Confidence", f"{prob * 100:.1f}%")
                        
                else:
                    st.error(f"Error from API: {r.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")

# ==========================================================
# VIEW: HOME
# ==========================================================
elif st.session_state.view == "home":
    st.title("🎬 Movie Recommender")
    st.markdown(
        "<div class='small-muted'>Type keyword → live TMDB search → open → details + recommendations</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    typed = st.text_input(
        "Search by movie title (keyword)", placeholder="Type: avenger, batman, love..."
    )

    st.divider()

    if typed.strip():
        st.markdown(f"### 🔎 Live TMDB Search for: *'{typed}'*")
        data, err = api_get_json("/tmdb/search", params={"query": typed.strip(), "page": 1})
        
        if err:
            st.error(f"Search API Error: {err}")
        elif not data or not data.get("results"):
            st.warning("No movies found on TMDB. Check your spelling!")
        else:
            suggestions, tmdb_cards = parse_tmdb_search_to_cards(data, typed, limit=12)
            if not tmdb_cards:
                st.warning("No matches could be rendered.")
            else:
                # Still hardcode columns to 6 since we removed the slider
                poster_grid(tmdb_cards, cols=6, key_prefix="tmdb_results")
        st.stop()

    st.markdown(f"### 🏠 {selected_category_label}")

    # Fetch 40 movies to ensure we have enough left after the rating filter drops the bad ones
    home_cards_raw, err = api_get_json(
        "/home", params={"category": home_category, "limit": 40}
    )
    if err or not home_cards_raw:
        st.error(f"Home feed failed: {err or 'Unknown error'}")
        st.stop()
        
    # Apply the new Minimum Rating filter before sending to the UI
    home_cards_filtered = [c for c in home_cards_raw if (c.get("vote_average") or 0.0) >= min_rating]

    # Limit to 24 results for clean UI
    poster_grid(home_cards_filtered[:24], cols=6, key_prefix="home_feed")

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
                cols=6,
                key_prefix="details_tfidf",
            )

            st.markdown("#### 🎭 More Like This (Genre)")
            poster_grid(
                bundle.get("genre_recommendations", []),
                cols=6,
                key_prefix="details_genre",
            )
        else:
            st.info("Showing Genre recommendations (fallback).")
            genre_only, err3 = api_get_json(
                "/recommend/genre", params={"tmdb_id": tmdb_id, "limit": 18}
            )
            if not err3 and genre_only:
                poster_grid(
                    genre_only, cols=6, key_prefix="details_genre_fallback"
                )
            else:
                st.warning("No recommendations available right now.")
    else:
        st.warning("No title available to compute recommendations.")