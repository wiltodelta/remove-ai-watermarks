(function () {
  "use strict";

  const PERMISSIVE_LICENSES = new Set(["apache-2.0", "mit", "bsd-3-clause"]);
  const DATE_FMT = new Intl.DateTimeFormat("en-CA", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });

  const SLICE_LABELS = [
    ["camera_photo", "Camera photos (50)"],
    ["ui_screenshot", "UI screenshots (50)"],
    ["science_diagram", "Science diagrams (50)"],
    ["scanned_document", "Scanned documents (50)"],
  ];

  const NATIVE_FAMILIES = [
    ["flux", "FLUX.1-dev (25)"],
    ["stable_diffusion", "SDXL 1.0 (25)"],
    ["openai", "GPT-Image-1 (25)"],
    ["google", "Gemini Nano Banana Pro (25)"],
    ["midjourney", "Midjourney v6.1 (25)"],
  ];

  const state = {
    payload: null,
    filterPermissive: false,
    filterZeroFpr: false,
    filterAttribution: false,
    sortBy: "default",
    sortAsc: false,
    expandedIds: new Set(["wiltodelta/raiw-source-classify"]),
  };

  function isPermissive(lic) {
    return PERMISSIVE_LICENSES.has(String(lic || "").toLowerCase());
  }

  function fprClass(fpr) {
    if (fpr <= 0.01) return "fpr-zero";
    if (fpr <= 5.0) return "fpr-mid";
    return "fpr-high";
  }

  function escapeHtml(val) {
    return String(val)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function modelHubUrl(modelId) {
    const baseRepo = modelId.split(" + ")[0].trim();
    return "https://huggingface.co/" + baseRepo;
  }

  function formatLocalDate(isoStr) {
    if (!isoStr) return "-";
    const dt = new Date(isoStr);
    return Number.isNaN(dt.getTime()) ? "-" : DATE_FMT.format(dt);
  }

  const SORT_VALUE = {
    fpr: (r) => r.non_ai_fpr_pct,
    native_recall: (r) => r.native_ai.recall_pct,
    native_attr: (r) => r.native_ai.provider_attribution_pct,
    arena_recall: (r) => r.arena_ai.recall_pct,
    precision: (r) => r.precision_pct,
    weights: (r) => r.weights_mb,
    latency: (r) => r.median_cpu_ms,
    updated: (r) => r.weights_updated_epoch || Date.parse(r.weights_updated_at) || 0,
  };

  function sortModels(rows) {
    const getter = SORT_VALUE[state.sortBy];
    if (!getter) {
      return rows.slice();
    }
    const dir = state.sortAsc ? 1 : -1;
    return rows.slice().sort((a, b) => {
      const va = getter(a);
      const vb = getter(b);
      if (va === vb) return a.non_ai_fpr_pct - b.non_ai_fpr_pct;
      return (va - vb) * dir;
    });
  }

  function filterModels(rows) {
    return rows.filter((r) => {
      if (state.filterPermissive && !isPermissive(r.license)) return false;
      if (state.filterZeroFpr && r.non_ai_fpr_pct > 0.0) return false;
      if (
        state.filterAttribution &&
        r.native_ai.provider_attribution_pct <= 0.0 &&
        r.arena_ai.provider_attribution_pct <= 0.0
      ) {
        return false;
      }
      return true;
    });
  }

  function renderDrawer(r) {
    const sf = r.slice_fpr_pct || {};
    const sliceBars = SLICE_LABELS.map(([key, label]) => {
      const val = Number(sf[key] || 0);
      const fillCls = val === 0 ? "bar-fill-pass" : "bar-fill-trap";
      const widthPct = val === 0 ? 2 : Math.min(100, val);
      const fpCount = Math.round((val / 100) * 50);
      return (
        '<div class="bar-item">' +
        '<span class="bar-label">' +
        escapeHtml(label) +
        "</span>" +
        '<div class="bar-track"><div class="bar-fill ' +
        fillCls +
        '" style="width:' +
        widthPct +
        '%"></div></div>' +
        '<span class="bar-value">' +
        val.toFixed(1) +
        "% (" +
        fpCount +
        "/50)</span>" +
        "</div>"
      );
    }).join("");

    const natProv = (r.native_ai && r.native_ai.per_provider) || {};
    const provBars = NATIVE_FAMILIES.map(([key, label]) => {
      const entry = natProv[key] || { recall_pct: 0, attribution_pct: 0 };
      const rec = Number(entry.recall_pct || 0);
      const attr = Number(entry.attribution_pct || 0);
      const barVal = attr > 0 ? attr : rec;
      const barCls = attr > 0 ? "bar-fill-cobalt" : "bar-fill-pass";
      const textVal =
        attr > 0
          ? attr.toFixed(0) + "% attr / " + rec.toFixed(0) + "% rec"
          : rec.toFixed(0) + "% binary";
      return (
        '<div class="bar-item">' +
        '<span class="bar-label">' +
        escapeHtml(label) +
        "</span>" +
        '<div class="bar-track"><div class="bar-fill ' +
        barCls +
        '" style="width:' +
        Math.min(100, barVal) +
        '%"></div></div>' +
        '<span class="bar-value">' +
        escapeHtml(textVal) +
        "</span>" +
        "</div>"
      );
    }).join("");

    return (
      '<tr class="drawer-row"><td colspan="11"><div class="drawer-inner">' +
      '<div class="drawer-panel">' +
      "<h3>False-positive breakdown on 200 real non-AI traps</h3>" +
      '<div class="bar-list">' +
      sliceBars +
      "</div>" +
      "</div>" +
      '<div class="drawer-panel">' +
      "<h3>Untouched native exports by generator family (125 files)</h3>" +
      '<div class="bar-list">' +
      provBars +
      "</div>" +
      "</div>" +
      '<div class="drawer-meta">' +
      "<span>Weights updated: <code>" +
      escapeHtml(formatLocalDate(r.weights_updated_at)) +
      "</code> (repo: <code>" +
      escapeHtml(formatLocalDate(r.repo_updated_at)) +
      "</code>)</span>" +
      "<span>Runtime: <code>" +
      escapeHtml(r.runtime) +
      "</code></span>" +
      "<span>Pinned revision: <code>" +
      escapeHtml(String(r.revision).slice(0, 21)) +
      "</code></span>" +
      '<span>Model card: <a href="' +
      escapeHtml(modelHubUrl(r.model_id)) +
      '" target="_blank" rel="noopener">' +
      escapeHtml(r.model_id.split(" + ")[0]) +
      "</a></span>" +
      "</div>" +
      "</div></td></tr>"
    );
  }

  function renderTable() {
    const tbody = document.getElementById("matrix-body");
    const countLabel = document.getElementById("visible-count");
    if (!state.payload || !tbody) return;

    const filtered = sortModels(filterModels(state.payload.results));
    countLabel.textContent =
      "Showing " +
      filtered.length +
      " of " +
      state.payload.results.length +
      " evaluated configurations";

    if (filtered.length === 0) {
      tbody.innerHTML =
        '<tr><td colspan="11" class="empty-state">No evaluated detectors match all active filters. Clear one of the filter buttons above to compare the full table.</td></tr>';
      return;
    }

    const html = [];
    filtered.forEach((r) => {
      const expanded = state.expandedIds.has(r.model_id);
      const licCls = isPermissive(r.license) ? "lic-permissive" : "lic-restricted";
      const sf = r.slice_fpr_pct || {};
      const quad =
        Math.round(sf.camera_photo || 0) +
        "% / " +
        Math.round(sf.ui_screenshot || 0) +
        "% / " +
        Math.round(sf.science_diagram || 0) +
        "% / " +
        Math.round(sf.scanned_document || 0) +
        "%";

      html.push(
        '<tr class="model-row" tabindex="0" role="button" aria-expanded="' +
          (expanded ? "true" : "false") +
          '" data-model="' +
          escapeHtml(r.model_id) +
          '">' +
          '<td class="col-left"><div class="model-cell">' +
          '<span class="expand-glyph" aria-hidden="true">' +
          (expanded ? "-" : "+") +
          "</span>" +
          '<span class="model-name">' +
          escapeHtml(r.model_id) +
          "</span>" +
          "</div></td>" +
          '<td class="col-left"><span class="license-pill ' +
          licCls +
          '">' +
          escapeHtml(r.license) +
          "</span></td>" +
          "<td>" +
          escapeHtml(formatLocalDate(r.weights_updated_at)) +
          "</td>" +
          "<td>" +
          r.weights_mb.toFixed(1) +
          " MB</td>" +
          "<td>" +
          Math.round(r.median_cpu_ms) +
          " ms</td>" +
          '<td><span class="metric-chip ' +
          fprClass(r.non_ai_fpr_pct) +
          '">' +
          r.non_ai_fpr_pct.toFixed(1) +
          "% (" +
          r.non_ai_fp +
          "/" +
          r.non_ai_total +
          ")</span></td>" +
          '<td class="slice-quad">' +
          quad +
          "</td>" +
          "<td>" +
          r.native_ai.recall_pct.toFixed(1) +
          "%</td>" +
          "<td>" +
          r.native_ai.provider_attribution_pct.toFixed(1) +
          "%</td>" +
          "<td>" +
          r.arena_ai.recall_pct.toFixed(1) +
          "%</td>" +
          "<td>" +
          r.precision_pct.toFixed(1) +
          "%</td>" +
          "</tr>"
      );

      if (expanded) {
        html.push(renderDrawer(r));
      }
    });

    tbody.innerHTML = html.join("");
  }

  function toggleRow(modelId) {
    if (state.expandedIds.has(modelId)) {
      state.expandedIds.delete(modelId);
    } else {
      state.expandedIds.add(modelId);
    }
    renderTable();
  }

  function wireEvents() {
    const tbody = document.getElementById("matrix-body");
    tbody.addEventListener("click", (ev) => {
      if (ev.target.closest("a")) return;
      const row = ev.target.closest(".model-row");
      if (row && row.dataset.model) {
        toggleRow(row.dataset.model);
      }
    });
    tbody.addEventListener("keydown", (ev) => {
      if (ev.key !== "Enter" && ev.key !== " ") return;
      const row = ev.target.closest(".model-row");
      if (row && row.dataset.model) {
        ev.preventDefault();
        toggleRow(row.dataset.model);
      }
    });

    const filterMap = [
      ["btn-permissive", "filterPermissive"],
      ["btn-zero-fpr", "filterZeroFpr"],
      ["btn-attribution", "filterAttribution"],
    ];
    filterMap.forEach(([id, key]) => {
      const btn = document.getElementById(id);
      if (!btn) return;
      btn.addEventListener("click", () => {
        state[key] = !state[key];
        btn.setAttribute("aria-pressed", state[key] ? "true" : "false");
        renderTable();
      });
    });

    const sortSelect = document.getElementById("sort-select");
    if (sortSelect) {
      sortSelect.addEventListener("change", () => {
        const val = sortSelect.value;
        if (val === "default") {
          state.sortBy = "default";
          state.sortAsc = false;
        } else {
          const parts = val.split(":");
          state.sortBy = parts[0];
          state.sortAsc = parts[1] === "asc";
        }
        renderTable();
      });
    }

    document.querySelectorAll("button.th-sort").forEach((btn) => {
      btn.addEventListener("click", () => {
        const col = btn.dataset.col;
        const defaultAsc = col === "fpr" || col === "weights" || col === "latency";
        if (state.sortBy === col) {
          state.sortAsc = !state.sortAsc;
        } else {
          state.sortBy = col;
          state.sortAsc = defaultAsc;
        }
        renderTable();
      });
    });

    const copyBtn = document.getElementById("copy-cmd-btn");
    if (copyBtn) {
      copyBtn.addEventListener("click", async () => {
        const codeEl = document.querySelector(".cmd-block code");
        const cmd = codeEl ? codeEl.textContent.trim() : "";
        try {
          await navigator.clipboard.writeText(cmd);
          copyBtn.textContent = "Copied command";
          setTimeout(() => {
            copyBtn.textContent = "Copy benchmark command";
          }, 1800);
        } catch (_err) {
          copyBtn.textContent = "Select command text";
        }
      });
    }
  }

  async function init() {
    wireEvents();
    const res = await fetch("./data/leaderboard.json");
    state.payload = await res.json();
    renderTable();
  }

  init();
})();
