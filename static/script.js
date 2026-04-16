const state = {
  category: "trending",
  limit: 24,
};

const elements = {
  homeButton: document.getElementById("homeButton"),
  backButton: document.getElementById("backButton"),
  refreshButton: document.getElementById("refreshButton"),
  searchButton: document.getElementById("searchButton"),
  searchInput: document.getElementById("searchInput"),
  categorySelect: document.getElementById("categorySelect"),
  limitSelect: document.getElementById("limitSelect"),
  suggestions: document.getElementById("suggestions"),
  statusBanner: document.getElementById("statusBanner"),
  homeTitle: document.getElementById("homeTitle"),
  homeSection: document.getElementById("homeSection"),
  detailsSection: document.getElementById("detailsSection"),
  homeGrid: document.getElementById("homeGrid"),
  detailsCard: document.getElementById("detailsCard"),
  tfidfGrid: document.getElementById("tfidfGrid"),
  genreGrid: document.getElementById("genreGrid"),
  template: document.getElementById("movieCardTemplate"),
};

function showStatus(message, isError = false) {
  elements.statusBanner.textContent = message;
  elements.statusBanner.classList.remove("hidden");
  elements.statusBanner.style.background = isError
    ? "rgba(255, 93, 93, 0.18)"
    : "rgba(255, 123, 84, 0.18)";
  elements.statusBanner.style.borderColor = isError
    ? "rgba(255, 93, 93, 0.28)"
    : "rgba(255, 123, 84, 0.24)";
}

function hideStatus() {
  elements.statusBanner.classList.add("hidden");
}

async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;

    try {
      const data = await response.json();
      if (typeof data.detail === "string" && data.detail.trim()) {
        message = data.detail.trim();
      }
    } catch (jsonError) {
      const text = await response.text();
      if (text.trim()) {
        message = text.trim();
      }
    }

    throw new Error(message);
  }
  return response.json();
}

function createMovieCard(movie) {
  const node = elements.template.content.firstElementChild.cloneNode(true);
  const posterButton = node.querySelector(".card-poster-button");
  const image = node.querySelector(".movie-poster");
  const title = node.querySelector(".movie-name");
  const meta = node.querySelector(".movie-meta");
  const detailButton = node.querySelector(".inline-button");

  image.src =
    movie.poster_url ||
    "https://via.placeholder.com/500x750/14223a/f5f7fb?text=No+Poster";
  image.alt = movie.title || "Movie poster";
  title.textContent = movie.title || "Untitled";

  const infoParts = [];
  if (movie.release_date) {
    infoParts.push(movie.release_date.slice(0, 4));
  }
  if (typeof movie.vote_average === "number") {
    infoParts.push(`Rating ${movie.vote_average.toFixed(1)}`);
  }
  meta.textContent = infoParts.join(" • ") || "Tap to view details";

  const open = () => {
    if (movie.tmdb_id) {
      loadDetails(movie.tmdb_id);
    } else if (movie.title) {
      loadLocalDetails(movie.title);
    }
  };

  posterButton.addEventListener("click", open);
  detailButton.addEventListener("click", open);

  return node;
}

function renderMovieGrid(container, movies, emptyMessage) {
  container.innerHTML = "";
  if (!movies.length) {
    container.innerHTML = `<p class="movie-meta">${emptyMessage}</p>`;
    return;
  }

  const fragment = document.createDocumentFragment();
  movies.forEach((movie) => fragment.appendChild(createMovieCard(movie)));
  container.appendChild(fragment);
}

function renderSuggestions(movies) {
  elements.suggestions.innerHTML = "";
  movies.slice(0, 6).forEach((movie) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "suggestion-chip";
    button.textContent = `${movie.title}${
      movie.release_date ? ` (${movie.release_date.slice(0, 4)})` : ""
    }`;
    button.addEventListener("click", () => loadDetails(movie.tmdb_id));
    elements.suggestions.appendChild(button);
  });
}

function showHome() {
  elements.homeSection.classList.remove("hidden");
  elements.detailsSection.classList.add("hidden");
}

function showDetails() {
  elements.homeSection.classList.add("hidden");
  elements.detailsSection.classList.remove("hidden");
}

async function loadHomeFeed() {
  hideStatus();
  elements.homeTitle.textContent = `${state.category.replaceAll("_", " ")} picks`;
  elements.homeGrid.innerHTML = `<p class="movie-meta">Loading movies...</p>`;

  try {
    const movies = await fetchJSON(
      `/home?category=${encodeURIComponent(state.category)}&limit=${state.limit}`
    );
    renderMovieGrid(
      elements.homeGrid,
      movies,
      "No movies are available for this category right now."
    );
    showHome();
  } catch (error) {
    try {
      const movies = await fetchJSON(`/offline/home?limit=${state.limit}`);
      renderMovieGrid(
        elements.homeGrid,
        movies,
        "No local movies are available right now."
      );
      elements.homeTitle.textContent = "Local picks";
      showHome();
      showStatus(
        "Live movie data is unavailable right now, so you're seeing recommendations from the local dataset instead."
      );
    } catch (fallbackError) {
      renderMovieGrid(elements.homeGrid, [], "Could not load the home feed.");
      showStatus(error.message, true);
    }
  }
}

async function performSearch() {
  const query = elements.searchInput.value.trim();
  elements.suggestions.innerHTML = "";

  if (query.length < 2) {
    showStatus("Type at least 2 characters to search for a movie.", true);
    return;
  }

  showStatus(`Searching for "${query}"...`);

  try {
    const data = await fetchJSON(`/tmdb/search?query=${encodeURIComponent(query)}`);
    const results = data.results || [];
    const mapped = results
      .filter((movie) => movie.id && (movie.title || movie.name))
      .map((movie) => ({
        tmdb_id: movie.id,
        title: movie.title || movie.name,
        poster_url: movie.poster_path
          ? `https://image.tmdb.org/t/p/w500${movie.poster_path}`
          : null,
        release_date: movie.release_date || "",
        vote_average: movie.vote_average,
      }));

    renderSuggestions(mapped);
    renderMovieGrid(
      elements.homeGrid,
      mapped,
      "No matching movies were found for your search."
    );
    elements.homeTitle.textContent = `Search results for "${query}"`;
    showHome();
    hideStatus();
  } catch (error) {
    try {
      const localResults = await fetchJSON(
        `/offline/search?query=${encodeURIComponent(query)}&limit=${state.limit}`
      );
      renderSuggestions(localResults);
      renderMovieGrid(
        elements.homeGrid,
        localResults,
        "No matching movies were found in the local dataset."
      );
      elements.homeTitle.textContent = `Local results for "${query}"`;
      showHome();
      showStatus(
        "TMDB search is unavailable right now, so these results are coming from the local dataset instead."
      );
    } catch (fallbackError) {
      showStatus(error.message, true);
    }
  }
}

function renderDetails(details) {
  const genres = (details.genres || []).map((genre) => genre.name).join(" • ");
  const releaseYear = details.release_date ? details.release_date.slice(0, 4) : "N/A";

  elements.detailsCard.innerHTML = `
    <div class="details-poster">
      <img
        src="${
          details.poster_url ||
          "https://via.placeholder.com/500x750/14223a/f5f7fb?text=No+Poster"
        }"
        alt="${details.title || "Movie"} poster"
      />
    </div>
    <div class="details-content glass-panel">
      <p class="section-kicker">Featured title</p>
      <h2 class="detail-title">${details.title || "Untitled"}</h2>
      <div class="meta-row">
        <span class="meta-pill">Released ${releaseYear}</span>
        ${
          genres
            ? `<span class="meta-pill">${genres}</span>`
            : `<span class="meta-pill">Genre unavailable</span>`
        }
      </div>
      <p class="detail-overview">${
        details.overview || "No overview is available for this movie yet."
      }</p>
      ${
        details.backdrop_url
          ? `<img class="backdrop-image" src="${details.backdrop_url}" alt="${details.title} backdrop" />`
          : ""
      }
    </div>
  `;
}

async function loadDetails(tmdbId) {
  showDetails();
  showStatus("Loading movie details and recommendations...");
  elements.detailsCard.innerHTML = `<p class="movie-meta">Loading details...</p>`;
  elements.tfidfGrid.innerHTML = "";
  elements.genreGrid.innerHTML = "";

  try {
    const details = await fetchJSON(`/movie/id/${tmdbId}`);
    renderDetails(details);

    try {
      const bundle = await fetchJSON(
        `/movie/search?query=${encodeURIComponent(details.title)}&tfidf_top_n=12&genre_limit=12`
      );
      const tfidfItems = (bundle.tfidf_recommendations || [])
        .map((item) => item.tmdb)
        .filter(Boolean);
      const genreItems = bundle.genre_recommendations || [];

      renderMovieGrid(
        elements.tfidfGrid,
        tfidfItems,
        "No TF-IDF recommendations were available for this title."
      );
      renderMovieGrid(
        elements.genreGrid,
        genreItems,
        "No genre recommendations were available."
      );
      hideStatus();
    } catch (bundleError) {
      try {
        const localBundle = await fetchJSON(
          `/offline/movie?title=${encodeURIComponent(details.title)}&tfidf_top_n=12&genre_limit=12`
        );
        renderMovieGrid(
          elements.tfidfGrid,
          localBundle.tfidf_recommendations || [],
          "No TF-IDF recommendations were available for this title."
        );
        renderMovieGrid(
          elements.genreGrid,
          localBundle.genre_recommendations || [],
          "No genre recommendations were available."
        );
        showStatus(
          "Movie details loaded. Recommendations are being shown from the local dataset because live TMDB matches are unavailable."
        );
      } catch (localError) {
        const genreItems = await fetchJSON(`/recommend/genre?tmdb_id=${tmdbId}&limit=12`);
        renderMovieGrid(
          elements.tfidfGrid,
          [],
          "No TF-IDF recommendations were available for this title."
        );
        renderMovieGrid(
          elements.genreGrid,
          genreItems,
          "No genre recommendations were available."
        );
        showStatus(
          "Movie details loaded. Showing genre recommendations while similar-title matches are unavailable."
        );
      }
    }
  } catch (error) {
    elements.detailsCard.innerHTML = `<p class="movie-meta">Could not load this movie.</p>`;
    showStatus(error.message, true);
  }
}

async function loadLocalDetails(title) {
  showDetails();
  showStatus("Loading recommendations from the local dataset...");
  elements.detailsCard.innerHTML = `<p class="movie-meta">Loading details...</p>`;
  elements.tfidfGrid.innerHTML = "";
  elements.genreGrid.innerHTML = "";

  try {
    const bundle = await fetchJSON(
      `/offline/movie?title=${encodeURIComponent(title)}&tfidf_top_n=12&genre_limit=12`
    );
    renderDetails(bundle.movie_details);
    renderMovieGrid(
      elements.tfidfGrid,
      bundle.tfidf_recommendations || [],
      "No TF-IDF recommendations were available for this title."
    );
    renderMovieGrid(
      elements.genreGrid,
      bundle.genre_recommendations || [],
      "No genre recommendations were available."
    );
    showStatus("Showing offline recommendations from the local movie dataset.");
  } catch (error) {
    elements.detailsCard.innerHTML = `<p class="movie-meta">Could not load this movie.</p>`;
    showStatus(error.message, true);
  }
}

elements.searchButton.addEventListener("click", performSearch);
elements.searchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    performSearch();
  }
});
elements.homeButton.addEventListener("click", loadHomeFeed);
elements.backButton.addEventListener("click", loadHomeFeed);
elements.refreshButton.addEventListener("click", () => {
  state.category = elements.categorySelect.value;
  state.limit = Number(elements.limitSelect.value);
  loadHomeFeed();
});
elements.categorySelect.addEventListener("change", () => {
  state.category = elements.categorySelect.value;
});
elements.limitSelect.addEventListener("change", () => {
  state.limit = Number(elements.limitSelect.value);
});

loadHomeFeed();
