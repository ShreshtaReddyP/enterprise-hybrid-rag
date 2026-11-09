// Frontend logic for DocuVault Enterprise Hybrid RAG

const queryForm = document.getElementById("queryForm");
const queryInput = document.getElementById("queryInput");
const submitBtn = document.getElementById("submitBtn");
const topKSelect = document.getElementById("topKSelect");

const answerContent = document.getElementById("answerContent");
const citationsSection = document.getElementById("citationsSection");
const citationsPills = document.getElementById("citationsPills");

const faithfulnessVal = document.getElementById("faithfulnessVal");
const relevanceVal = document.getElementById("relevanceVal");
const precisionVal = document.getElementById("precisionVal");
const latencyVal = document.getElementById("latencyVal");
const latencyDetail = document.getElementById("latencyDetail");
const chunksList = document.getElementById("chunksList");

const totalChunksBadge = document.getElementById("totalChunksBadge");
const docList = document.getElementById("docList");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");

document.addEventListener("DOMContentLoaded", () => {
  loadDocumentStats();
  setupEventListeners();
});

function setupEventListeners() {
  queryForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleSearch();
  });

  // Dropzone click & drag
  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", handleFileUpload);

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "var(--accent-cyan)";
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "var(--border-color)";
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "var(--border-color)";
    if (e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
      handleFileUpload();
    }
  });
}

async function loadDocumentStats() {
  try {
    const res = await fetch("/api/documents");
    if (!res.ok) return;
    const data = await res.json();

    totalChunksBadge.textContent = `${data.total_chunks} Chunks Indexed`;

    docList.innerHTML = "";
    data.documents.forEach((doc) => {
      const card = document.createElement("div");
      card.className = "doc-card";
      card.innerHTML = `
        <div class="doc-card-title">📄 ${escapeHtml(doc.filename)}</div>
        <div class="doc-card-meta">
          <span>${doc.total_pages} Pages</span>
          <span>${doc.total_chunks} Chunks</span>
        </div>
      `;
      docList.appendChild(card);
    });
  } catch (err) {
    console.error("Failed to load documents:", err);
  }
}

async function handleFileUpload() {
  const file = fileInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  const prevText = dropzone.querySelector("p").innerHTML;
  dropzone.querySelector("p").textContent = `Indexing ${file.name}...`;

  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Upload failed");
    }

    alert(`Successfully parsed and indexed: ${file.name}`);
    await loadDocumentStats();
  } catch (err) {
    alert(`Upload Error: ${err.message}`);
  } finally {
    dropzone.querySelector("p").innerHTML = prevText;
    fileInput.value = "";
  }
}

async function handleSearch() {
  const query = queryInput.value.trim();
  if (!query) return;

  submitBtn.disabled = true;
  submitBtn.querySelector("span").textContent = "Retrieving...";
  answerContent.innerHTML = `<div class="loading-state">🔍 Executing Hybrid Retrieval (BM25 + Dense Embeddings) and Ragas Evaluation...</div>`;
  citationsSection.style.display = "none";

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        top_k: parseInt(topKSelect.value, 10),
        include_evaluation: true
      })
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Query failed");
    }

    const data = await res.json();
    renderResponse(data);
  } catch (err) {
    answerContent.innerHTML = `<p style="color: #ef4444;">⚠️ <strong>Query Error:</strong> ${escapeHtml(err.message)}</p>`;
  } finally {
    submitBtn.disabled = false;
    submitBtn.querySelector("span").textContent = "Analyze";
  }
}

function renderResponse(data) {
  // 1. Render Grounded Answer
  answerContent.innerHTML = formatMarkdown(data.answer);

  // 2. Render Citations Pills
  if (data.citations && data.citations.length > 0) {
    citationsSection.style.display = "block";
    citationsPills.innerHTML = "";
    data.citations.forEach((c) => {
      const pill = document.createElement("button");
      pill.className = "citation-pill";
      pill.innerHTML = `[${c.citation_id}] ${c.source_file} (p. ${c.page_number})`;
      pill.onclick = () => highlightChunk(c.citation_id - 1);
      citationsPills.appendChild(pill);
    });
  }

  // 3. Render Evaluation Metrics
  if (data.evaluation) {
    const evalData = data.evaluation;
    faithfulnessVal.textContent = `${Math.round(evalData.faithfulness_score * 100)}%`;
    relevanceVal.textContent = `${Math.round(evalData.answer_relevance_score * 100)}%`;
    precisionVal.textContent = `${Math.round(evalData.context_precision * 100)}%`;

    const lat = evalData.latency;
    latencyVal.textContent = `${Math.round(lat.total_ms)} ms`;
    latencyDetail.textContent = `Ret: ${Math.round(lat.retrieval_ms)}ms | Rerank: ${Math.round(lat.rerank_ms)}ms | Gen: ${Math.round(lat.generation_ms)}ms`;

    const auditBadge = document.getElementById("auditBadge");
    auditBadge.textContent = evalData.verdict.split(" ")[0] + " Confidence";
    if (evalData.faithfulness_score >= 0.85) {
      auditBadge.style.color = "var(--accent-emerald)";
      auditBadge.style.borderColor = "var(--accent-emerald)";
    }
  }

  // 4. Render Provenance Chunks List
  chunksList.innerHTML = "";
  data.retrieved_chunks.forEach((chunk, idx) => {
    const card = document.createElement("div");
    card.id = `chunk-card-${idx}`;
    card.className = "chunk-inspect-card";
    card.innerHTML = `
      <div class="chunk-meta-bar">
        <span>#${idx + 1} • ${chunk.source_file} (Page ${chunk.page_number})</span>
        <span style="color: var(--accent-cyan);">${chunk.chunk_id}</span>
      </div>
      <div class="chunk-snippet-text">${escapeHtml(chunk.text)}</div>
      <div class="score-tags">
        <span class="score-tag tag-dense">Dense: ${chunk.dense_score}</span>
        <span class="score-tag tag-bm25">BM25: ${chunk.sparse_score}</span>
        <span class="score-tag tag-rrf">RRF Rank: ${chunk.rrf_score}</span>
      </div>
    `;
    chunksList.appendChild(card);
  });
}

function highlightChunk(idx) {
  const card = document.getElementById(`chunk-card-${idx}`);
  if (card) {
    document.querySelectorAll(".chunk-inspect-card").forEach((c) => c.classList.remove("active-target"));
    card.classList.add("active-target");
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

window.setQuery = function(text) {
  queryInput.value = text;
  handleSearch();
};

window.runBenchmark = async function() {
  const modal = document.getElementById("benchmarkModal");
  const modalBody = document.getElementById("benchmarkBody");
  modal.style.display = "flex";
  modalBody.innerHTML = `<div class="loading-state">⏳ Executing automated Ragas audit benchmark across 3 test scenarios...</div>`;

  try {
    const res = await fetch("/api/benchmark");
    const data = await res.json();

    let tableHtml = `
      <div style="margin-bottom: 1rem; font-size: 0.85rem; color: #a7f3d0;">
        <strong>Benchmark Result:</strong> Mean Faithfulness: <strong>${Math.round(data.mean_faithfulness * 100)}%</strong> | 
        Mean Relevance: <strong>${Math.round(data.mean_relevance * 100)}%</strong>
      </div>
      <table style="width: 100%; border-collapse: collapse; font-size: 0.75rem;">
        <thead>
          <tr style="border-bottom: 1px solid var(--border-color); text-align: left; color: var(--text-secondary);">
            <th style="padding: 0.5rem;">Audit Query</th>
            <th style="padding: 0.5rem;">Faithfulness</th>
            <th style="padding: 0.5rem;">Relevance</th>
            <th style="padding: 0.5rem;">Verdict</th>
          </tr>
        </thead>
        <tbody>
    `;

    data.results.forEach((r) => {
      tableHtml += `
        <tr style="border-bottom: 1px solid rgba(55, 65, 81, 0.5);">
          <td style="padding: 0.5rem; color: #e5e7eb;">${r.query}</td>
          <td style="padding: 0.5rem; font-family: var(--font-mono); color: var(--accent-cyan);">${Math.round(r.faithfulness * 100)}%</td>
          <td style="padding: 0.5rem; font-family: var(--font-mono); color: var(--accent-emerald);">${Math.round(r.answer_relevance * 100)}%</td>
          <td style="padding: 0.5rem; font-size: 0.7rem; color: #a7f3d0;">${r.verdict.split("(")[0]}</td>
        </tr>
      `;
    });

    tableHtml += `</tbody></table>`;
    modalBody.innerHTML = tableHtml;
  } catch (err) {
    modalBody.innerHTML = `<p style="color: #ef4444;">Failed to run benchmark: ${err.message}</p>`;
  }
};

window.closeBenchmark = function() {
  document.getElementById("benchmarkModal").style.display = "none";
};

function formatMarkdown(text) {
  if (!text) return "";
  let formatted = escapeHtml(text);
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  formatted = formatted.replace(/\*(.*?)\*/g, "<em>$1</em>");
  formatted = formatted.replace(/`([^`]+)`/g, "<code style='background: #1f2937; padding: 0.15rem 0.35rem; border-radius: 4px; font-family: var(--font-mono); font-size: 0.85em;'>$1</code>");

  const lines = formatted.split("\n");
  let inList = false;
  let result = [];

  for (let line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      if (!inList) {
        result.push("<ul>");
        inList = true;
      }
      result.push(`<li>${trimmed.substring(2)}</li>`);
    } else {
      if (inList) {
        result.push("</ul>");
        inList = false;
      }
      if (trimmed.length > 0) {
        result.push(`<p>${line}</p>`);
      }
    }
  }
  if (inList) result.push("</ul>");
  return result.join("");
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
