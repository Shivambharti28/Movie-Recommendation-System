const state = {
  bootstrap: null,
  selectedGenres: new Set(),
  activeMovieId: null,
};

const heroEl = document.querySelector("#hero");
const resultsGridEl = document.querySelector("#resultsGrid");
const resultsTitleEl = document.querySelector("#resultsTitle");
const statusTextEl = document.querySelector("#statusText");
const railSectionsEl = document.querySelector("#railSections");
const genreChipsEl = document.querySelector("#genreChips");
const preferenceInputEl = document.querySelector("#preferenceInput");
const discoverButtonEl = document.querySelector("#discoverButton");
const refreshButtonEl = document.querySelector("#refreshButton");
const cardTemplate = document.querySelector("#movieCardTemplate");

const LANGUAGE_LABELS = {
  ar: "Arabic",
  bn: "Bengali",
  de: "German",
  en: "English",
  es: "Spanish",
  fr: "French",
  hi: "Hindi",
  it: "Italian",
  ja: "Japanese",
  kn: "Kannada",
  ko: "Korean",
  ml: "Malayalam",
  mr: "Marathi",
  pt: "Portuguese",
  ru: "Russian",
  ta: "Tamil",
  te: "Telugu",
  th: "Thai",
  tr: "Turkish",
  ur: "Urdu",
  zh: "Chinese",
};

async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function createPill(label, active = false) {
  const element = document.createElement("button");
  element.className = `pill${active ? " active" : ""}`;
  element.textContent = label;
  return element;
}

function renderHero(movie) {
  if (!movie) {
    heroEl.classList.add("loading");
    heroEl.innerHTML = "";
    return;
  }

  heroEl.classList.remove("loading");
  heroEl.style.setProperty("--hero-bg", `url(${movie.backdrop_url})`);
  heroEl.style.setProperty("background-image", `linear-gradient(180deg, rgba(4,8,14,0.05), rgba(4,9,15,0.95)), url(${movie.backdrop_url})`);
  heroEl.innerHTML = `
    <div class="hero-inner">
      <p class="eyebrow">Featured intelligence pick</p>
      <h2>${movie.title}</h2>
      <p>${movie.tagline || movie.overview || "Handpicked from the live TMDB catalog."}</p>
      <div class="hero-metrics">
        <span class="pill">${movie.release_year || "N/A"}</span>
        <span class="pill">${movie.vote_average} rating</span>
        <span class="pill">${movie.runtime ? `${movie.runtime} min` : "Feature film"}</span>
        ${(movie.genres || []).slice(0, 3).map((genre) => `<span class="pill">${genre}</span>`).join("")}
      </div>
      <div class="hero-metrics">
        <button class="ghost-button" id="heroRecommend">Generate Similar</button>
        ${movie.trailer_key ? `<a class="ghost-button" href="https://www.youtube.com/watch?v=${movie.trailer_key}" target="_blank" rel="noreferrer">Watch Trailer</a>` : ""}
      </div>
    </div>
  `;

  const recommendButton = document.querySelector("#heroRecommend");
  if (recommendButton) {
    recommendButton.addEventListener("click", () => loadRecommendations({ movieId: movie.id, title: `Because you liked ${movie.title}` }));
  }
}

function buildMovieCard(movie) {
  const fragment = cardTemplate.content.cloneNode(true);
  const card = fragment.querySelector(".movie-card");
  const image = fragment.querySelector(".poster");
  const title = fragment.querySelector(".card-title");
  const overview = fragment.querySelector(".card-overview");
  const year = fragment.querySelector(".card-year");
  const rating = fragment.querySelector(".card-rating");
  const genres = fragment.querySelector(".card-genres");
  const similarButton = fragment.querySelector(".card-recommend");
  const trailerButton = fragment.querySelector(".card-trailer");

  image.src = movie.poster_url || movie.backdrop_url || "";
  image.alt = movie.title;
  title.textContent = movie.title;
  overview.textContent = movie.overview || "No overview available yet.";
  year.textContent = movie.release_year || "N/A";
  rating.textContent = `${movie.vote_average || "-"} ★`;

  (movie.genres || []).slice(0, 3).forEach((genre) => {
    const pill = document.createElement("span");
    pill.className = "pill";
    pill.textContent = genre;
    genres.appendChild(pill);
  });

  similarButton.addEventListener("click", () => {
    loadRecommendations({ movieId: movie.id, title: `Similar to ${movie.title}` });
  });

  if (movie.trailer_key) {
    trailerButton.href = `https://www.youtube.com/watch?v=${movie.trailer_key}`;
  } else {
    trailerButton.remove();
  }

  return card;
}

function emptyResultsMessage() {
  resultsGridEl.className = "results-grid";
  resultsGridEl.innerHTML = "";
  resultsGridEl.innerHTML = `<div class="glass-panel" style="padding: 24px; border-radius: 24px; border: 1px solid rgba(255,255,255,0.08);">No close matches yet. Try a different mood, director, genre, or plot description.</div>`;
}

function renderGroupedResults({ sameLanguage = [], crossLanguage = [], anchorLanguageLabel = "Original" } = {}) {
  resultsGridEl.className = "results-group-stack";
  resultsGridEl.innerHTML = "";
  if (!sameLanguage.length && !crossLanguage.length) {
    emptyResultsMessage();
    return;
  }

  const groups = [
    {
      title: `${anchorLanguageLabel} Similar Movies`,
      subtitle: "Same-language matches first.",
      movies: sameLanguage,
    },
    {
      title: "Other Languages You May Like",
      subtitle: "Cross-language recommendations after the same-language set.",
      movies: crossLanguage,
    },
  ].filter((group) => group.movies.length);

  groups.forEach((group) => {
    const section = document.createElement("section");
    section.className = "results-group";
    section.innerHTML = `
      <div class="results-group-header">
        <div>
          <p class="eyebrow">Language Split</p>
          <h3>${group.title}</h3>
        </div>
        <p class="results-group-meta">${group.subtitle}</p>
      </div>
    `;

    const grid = document.createElement("div");
    grid.className = "group-grid";
    group.movies.forEach((movie) => grid.appendChild(buildMovieCard(movie)));
    section.appendChild(grid);
    resultsGridEl.appendChild(section);
  });
}

function renderResults(movies, options = {}) {
  if (options.grouped) {
    renderGroupedResults(options);
    return;
  }

  resultsGridEl.className = "results-grid";
  resultsGridEl.innerHTML = "";
  if (!movies.length) {
    emptyResultsMessage();
    return;
  }
  movies.forEach((movie) => resultsGridEl.appendChild(buildMovieCard(movie)));
}

function renderRails(sections) {
  railSectionsEl.innerHTML = "";
  const sectionLabels = {
    trending: "Trending Now",
    bollywood: "Bollywood Now",
    popular: "Popular Right Now",
    top_rated: "Top Rated Stories",
    upcoming: "Coming Soon",
  };

  Object.entries(sections || {}).forEach(([key, movies]) => {
    if (!movies.length) return;

    const rail = document.createElement("section");
    rail.className = "rail";
    rail.innerHTML = `
      <div class="section-header">
        <div>
          <p class="eyebrow">Live TMDB</p>
          <h2>${sectionLabels[key] || key}</h2>
        </div>
      </div>
    `;

    const track = document.createElement("div");
    track.className = "rail-track";
    movies.forEach((movie) => track.appendChild(buildMovieCard(movie)));
    rail.appendChild(track);
    railSectionsEl.appendChild(rail);
  });
}

function renderGenreChips(genres) {
  genreChipsEl.innerHTML = "";
  genres.forEach((genre) => {
    const chip = createPill(genre, state.selectedGenres.has(genre));
    chip.addEventListener("click", () => {
      if (state.selectedGenres.has(genre)) {
        state.selectedGenres.delete(genre);
      } else {
        state.selectedGenres.add(genre);
      }
      renderGenreChips(genres);
    });
    genreChipsEl.appendChild(chip);
  });

  if (state.selectedGenres.size > 0) {
    const clearChip = document.createElement("button");
    clearChip.className = "pill clear-pill";
    clearChip.textContent = "Clear Filters";
    clearChip.addEventListener("click", () => {
      state.selectedGenres.clear();
      renderGenreChips(genres);
    });
    genreChipsEl.appendChild(clearChip);
  }
}

async function loadBootstrap() {
  statusTextEl.textContent = "Loading catalog...";
  const payload = await fetchJSON("/api/bootstrap");
  state.bootstrap = payload;
  renderHero(payload.hero);
  renderRails(payload.sections);
  renderGenreChips(payload.genres || []);
  renderResults((payload.sections && payload.sections.trending) || []);

  const refreshTime = payload.stats?.last_refresh
    ? new Date(payload.stats.last_refresh * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : "just now";
  statusTextEl.textContent = payload.status?.error
    ? payload.status.error
    : `Catalog ready with ${payload.stats?.catalog_size || 0} titles. Refreshed at ${refreshTime}.`;
}

async function loadRecommendations({ movieId = null, title = "Curated for you", query = "" } = {}) {
  resultsTitleEl.textContent = title;
  statusTextEl.textContent = "Generating recommendations...";
  const selectedGenres = [...state.selectedGenres].join(",");
  const params = new URLSearchParams();
  if (movieId) params.set("movie_id", movieId);
  if (query) params.set("q", query);
  if (selectedGenres) params.set("genres", selectedGenres);

  const payload = await fetchJSON(`/api/recommend?${params.toString()}`);
  const breakdown = payload.language_breakdown || {};
  const sameLanguageResults = payload.same_language_results || [];
  const crossLanguageResults = payload.cross_language_results || [];

  renderResults(payload.results || [], {
    grouped: sameLanguageResults.length || crossLanguageResults.length,
    sameLanguage: sameLanguageResults,
    crossLanguage: crossLanguageResults,
    anchorLanguageLabel: breakdown.anchor_language_label || "Original",
  });

  if (sameLanguageResults.length || crossLanguageResults.length) {
    statusTextEl.textContent = `${sameLanguageResults.length} ${breakdown.anchor_language_label || "same-language"} recommendations first, then ${crossLanguageResults.length} cross-language recommendations.`;
    return;
  }

  statusTextEl.textContent = `${payload.results?.length || 0} recommendations ranked by semantic similarity, metadata match, and quality signals.`;
}

async function discoverMovies(query) {
  resultsTitleEl.textContent = query ? `Results for "${query}"` : "Curated for you";
  statusTextEl.textContent = "Finding the best movie match...";
  const selectedGenresArray = [...state.selectedGenres];
  const selectedGenres = selectedGenresArray.join(",");
  const params = new URLSearchParams();
  if (query) params.set("q", query);
  if (selectedGenres) params.set("genres", selectedGenres);

  const payload = await fetchJSON(`/api/discover?${params.toString()}`);
  const breakdown = payload.language_breakdown || {};
  const sameLanguageResults = payload.same_language_results || [];
  const crossLanguageResults = payload.cross_language_results || [];
  renderResults(payload.results || [], {
    grouped: sameLanguageResults.length || crossLanguageResults.length,
    sameLanguage: sameLanguageResults,
    crossLanguage: crossLanguageResults,
    anchorLanguageLabel: breakdown.anchor_language_label || "Original",
  });

  if (payload.mode === "collection_match") {
    if (payload.anchor) {
      renderHero(payload.anchor);
    }
    resultsTitleEl.textContent = payload.collection_label || `Results for "${query}"`;
    statusTextEl.textContent = payload.collection_description
      ? `${payload.results?.length || 0} titles. ${payload.collection_description}`
      : `${payload.results?.length || 0} titles from live collection discovery.`;
    return;
  }

  if (payload.anchor) {
    renderHero(payload.anchor);
    resultsTitleEl.textContent = `Similar to "${payload.anchor.title}"`;
    if (sameLanguageResults.length || crossLanguageResults.length) {
      statusTextEl.textContent = selectedGenresArray.length
        ? `${sameLanguageResults.length} ${breakdown.anchor_language_label || "same-language"} recommendations first, then ${crossLanguageResults.length} cross-language picks, with active filters: ${selectedGenresArray.join(", ")}.`
        : `${sameLanguageResults.length} ${breakdown.anchor_language_label || "same-language"} recommendations first, then ${crossLanguageResults.length} cross-language picks.`;
    } else {
      statusTextEl.textContent = selectedGenresArray.length
        ? `${payload.results?.length || 0} recommendations based on "${payload.anchor.title}", boosted by active filters: ${selectedGenresArray.join(", ")}.`
        : `${payload.results?.length || 0} recommendations based on an exact title match and semantic similarity.`;
    }
    return;
  }

  resultsTitleEl.textContent = query ? `Results for "${query}"` : "Genre-guided picks";
  statusTextEl.textContent = selectedGenresArray.length
    ? `${payload.results?.length || 0} recommendations ranked by semantic similarity, metadata match, and active filters: ${selectedGenresArray.join(", ")}.`
    : `${payload.results?.length || 0} recommendations ranked by semantic similarity, metadata match, and quality signals.`;
}

discoverButtonEl.addEventListener("click", async () => {
  const query = preferenceInputEl.value.trim();
  if (!query && state.selectedGenres.size === 0) {
    statusTextEl.textContent = "Add a mood, plot, or a few genres to start the AI match.";
    return;
  }
  await discoverMovies(query);
});

preferenceInputEl.addEventListener("keydown", async (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    discoverButtonEl.click();
  }
});

refreshButtonEl.addEventListener("click", async () => {
  statusTextEl.textContent = "Refreshing TMDB catalog...";
  await fetchJSON("/api/refresh");
  await loadBootstrap();
});

loadBootstrap().catch((error) => {
  statusTextEl.textContent = error.message;
});
