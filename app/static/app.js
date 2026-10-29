/**
 * Smart Notes Assistant - Frontend Controller
 */
document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const noteForm = document.getElementById("note-form");
  const noteTextInput = document.getElementById("note-text");
  const charCount = document.getElementById("char-count");
  const btnSaveNote = document.getElementById("btn-save-note");
  const notesContainer = document.getElementById("notes-container");
  const totalNotesPill = document.getElementById("total-notes-pill");
  const btnLoadStarter = document.getElementById("btn-load-starter");
  const filterTabs = document.querySelectorAll(".filter-tab");
  const searchInput = document.getElementById("search-input");
  const btnSearch = document.getElementById("btn-search");
  const btnClearSearch = document.getElementById("btn-clear-search");
  const searchResultsSection = document.getElementById("search-results-section");
  const searchResultsList = document.getElementById("search-results-list");
  const searchCount = document.getElementById("search-count");
  const btnCloseSearch = document.getElementById("btn-close-search");
  const toastEl = document.getElementById("toast");

  let currentCategory = "all";
  let currentUncertainOnly = false;

  // ---------------------------------------------------------------------------
  // Utility: Toast Notifications
  // ---------------------------------------------------------------------------
  function showToast(message, type = "success") {
    toastEl.textContent = message;
    toastEl.className = `toast ${type}`;
    setTimeout(() => {
      toastEl.className = "toast hidden";
    }, 3500);
  }

  // ---------------------------------------------------------------------------
  // Character Counter
  // ---------------------------------------------------------------------------
  noteTextInput.addEventListener("input", () => {
    charCount.textContent = noteTextInput.value.length;
  });

  // ---------------------------------------------------------------------------
  // Note Card Renderer
  // ---------------------------------------------------------------------------
  function createNoteElement(note, isSearchResult = false) {
    const el = document.createElement("div");
    el.className = "note-item";
    el.id = `note-${note.id}`;

    const uncertainBadge = note.uncertain
      ? `<span class="badge badge-uncertain" title="Model confidence is low or note is out-of-scope">⚠️ Uncertain</span>`
      : "";

    const scoreBadge = isSearchResult && note.score !== undefined
      ? `<span class="score-badge" title="Relevance Score">Relevance: ${Math.round(note.score * 100)}%</span>`
      : "";

    el.innerHTML = `
      <div class="note-top">
        <span class="note-id">${note.id}</span>
        <div class="badges-group">
          ${scoreBadge}
          <span class="badge badge-${note.category}">${note.category}</span>
          ${uncertainBadge}
        </div>
      </div>
      <div class="note-text">${escapeHtml(note.text)}</div>
      <div class="note-footer">
        <span>${note.created_at ? new Date(note.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}</span>
        <button class="delete-btn" data-id="${note.id}" title="Delete note">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="3 6 5 6 21 6"></polyline>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
          </svg>
        </button>
      </div>
    `;

    const delBtn = el.querySelector(".delete-btn");
    delBtn.addEventListener("click", () => deleteNote(note.id));

    return el;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // ---------------------------------------------------------------------------
  // Fetch & Render Notes
  // ---------------------------------------------------------------------------
  async function loadNotes() {
    try {
      let url = "/api/notes?";
      if (currentCategory && currentCategory !== "all") {
        url += `category=${encodeURIComponent(currentCategory)}&`;
      }
      if (currentUncertainOnly) {
        url += `uncertain_only=true&`;
      }

      const res = await fetch(url);
      if (!res.ok) throw new Error("Failed to load notes");
      const data = await res.json();

      notesContainer.innerHTML = "";
      totalNotesPill.textContent = `${data.total} notes`;

      if (data.notes.length === 0) {
        notesContainer.innerHTML = `<div class="empty-state">No notes found under this filter.</div>`;
        return;
      }

      data.notes.forEach((note) => {
        notesContainer.appendChild(createNoteElement(note));
      });
    } catch (err) {
      notesContainer.innerHTML = `<div class="empty-state">Error loading notes: ${err.message}</div>`;
    }
  }

  // ---------------------------------------------------------------------------
  // Filter Tabs
  // ---------------------------------------------------------------------------
  filterTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      filterTabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      if (tab.dataset.uncertain === "true") {
        currentCategory = "all";
        currentUncertainOnly = true;
      } else {
        currentCategory = tab.dataset.category;
        currentUncertainOnly = false;
      }

      loadNotes();
    });
  });

  // ---------------------------------------------------------------------------
  // Save Note Form Submission
  // ---------------------------------------------------------------------------
  noteForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = noteTextInput.value.trim();
    if (!text) return;

    btnSaveNote.disabled = true;
    btnSaveNote.textContent = "Classifying...";

    try {
      const res = await fetch("/api/notes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Failed to save note");
      }

      const savedNote = await res.json();
      noteTextInput.value = "";
      charCount.textContent = "0";

      const statusText = savedNote.uncertain
        ? `Saved as ${savedNote.category} (⚠️ Marked Uncertain)`
        : `Saved under ${savedNote.category} (${Math.round(savedNote.confidence * 100)}% confidence)`;

      showToast(statusText, "success");
      loadNotes();
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      btnSaveNote.disabled = false;
      btnSaveNote.textContent = "Save & Classify";
    }
  });

  // ---------------------------------------------------------------------------
  // Delete Note
  // ---------------------------------------------------------------------------
  async function deleteNote(id) {
    if (!confirm(`Delete note ${id}?`)) return;

    try {
      const res = await fetch(`/api/notes/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Failed to delete note");
      showToast(`Deleted note ${id}`, "success");
      loadNotes();
    } catch (err) {
      showToast(err.message, "error");
    }
  }

  // ---------------------------------------------------------------------------
  // Load Starter Notes Button
  // ---------------------------------------------------------------------------
  btnLoadStarter.addEventListener("click", async () => {
    btnLoadStarter.disabled = true;
    try {
      const res = await fetch("/api/notes/load-starter", { method: "POST" });
      if (!res.ok) throw new Error("Failed to load starter notes");
      const data = await res.json();
      showToast(data.message, "success");
      loadNotes();
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      btnLoadStarter.disabled = false;
    }
  });

  // ---------------------------------------------------------------------------
  // Semantic Search
  // ---------------------------------------------------------------------------
  async function executeSearch() {
    const q = searchInput.value.trim();
    if (!q) {
      closeSearch();
      return;
    }

    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
      if (!res.ok) throw new Error("Search request failed");
      const data = await res.json();

      searchResultsSection.classList.remove("hidden");
      searchResultsList.innerHTML = "";
      searchCount.textContent = data.count;

      if (data.results.length === 0) {
        searchResultsList.innerHTML = `
          <div class="empty-state">
            No matching notes found for "${escapeHtml(q)}".<br>
            <span style="font-size:0.8rem; color:var(--text-dim);">Low relevance scores filtered to prevent false matches.</span>
          </div>`;
        return;
      }

      data.results.forEach((result) => {
        searchResultsList.appendChild(createNoteElement(result, true));
      });
    } catch (err) {
      showToast(err.message, "error");
    }
  }

  btnSearch.addEventListener("click", executeSearch);
  searchInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") executeSearch();
  });

  searchInput.addEventListener("input", () => {
    btnClearSearch.classList.toggle("hidden", !searchInput.value.trim());
  });

  btnClearSearch.addEventListener("click", () => {
    searchInput.value = "";
    btnClearSearch.classList.add("hidden");
    closeSearch();
  });

  function closeSearch() {
    searchResultsSection.classList.add("hidden");
    searchResultsList.innerHTML = "";
  }

  btnCloseSearch.addEventListener("click", closeSearch);

  // Initial load
  loadNotes();
});
