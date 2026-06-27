const EXAMPLES = Array.isArray(window.MIRRORPPR_EXAMPLES) ? window.MIRRORPPR_EXAMPLES : [];
const QUALITATIVE_CASES = Array.isArray(window.MIRRORPPR_QUALITATIVE) ? window.MIRRORPPR_QUALITATIVE : [];
const PREFERRED_EXAMPLE_IDS = {
  simulated: "face-034",
};

const state = {
  subset: "simulated",
  index: 0,
  edited: false,
  expanded: false,
};

const qualitativeState = {
  subset: "simulated",
  index: 0,
  edited: false,
  expanded: false,
  baseline: "",
};

const ZOOM_STEP = 1.55;
const WHEEL_ZOOM_STEP = 1.16;
const MAX_ZOOM = 5;

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function currentExamples() {
  const preferredId = PREFERRED_EXAMPLE_IDS[state.subset];
  const examples = EXAMPLES.filter((example) => example.subsetKey === state.subset);
  if (!preferredId) return examples;

  return examples.slice().sort((a, b) => {
    if (a.id === preferredId) return -1;
    if (b.id === preferredId) return 1;
    return 0;
  });
}

function exampleImageCard(beforeSrc, afterSrc, beforeLabel, afterLabel, altText, extraHtml = "") {
  return `
    <figure class="example-image-card">
      <img
        class="example-image"
        src="${beforeSrc}"
        data-before-src="${beforeSrc}"
        data-after-src="${afterSrc}"
        alt="${escapeHtml(altText)}"
        draggable="false"
      >
      ${extraHtml}
      <figcaption
        class="image-state-label"
        data-before-label="${escapeHtml(beforeLabel)}"
        data-after-label="${escapeHtml(afterLabel)}"
      >${escapeHtml(beforeLabel)}</figcaption>
    </figure>
  `;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function getCardZoom(card) {
  return {
    scale: Number(card.dataset.zoomScale || "1"),
    panX: Number(card.dataset.zoomPanX || "0"),
    panY: Number(card.dataset.zoomPanY || "0"),
    originX: Number(card.dataset.zoomOriginX || "50"),
    originY: Number(card.dataset.zoomOriginY || "50"),
  };
}

function setCardZoom(card, zoom) {
  const scale = clamp(Number(zoom.scale || 1), 1, MAX_ZOOM);
  const panX = Number(zoom.panX || 0);
  const panY = Number(zoom.panY || 0);
  const originX = clamp(Number(zoom.originX || 50), 0, 100);
  const originY = clamp(Number(zoom.originY || 50), 0, 100);

  card.dataset.zoomScale = scale.toFixed(4);
  card.dataset.zoomPanX = panX.toFixed(2);
  card.dataset.zoomPanY = panY.toFixed(2);
  card.dataset.zoomOriginX = originX.toFixed(2);
  card.dataset.zoomOriginY = originY.toFixed(2);
  card.style.setProperty("--zoom-scale", scale.toFixed(4));
  card.style.setProperty("--zoom-pan-x", `${panX.toFixed(2)}px`);
  card.style.setProperty("--zoom-pan-y", `${panY.toFixed(2)}px`);
  card.style.setProperty("--zoom-x", `${originX.toFixed(2)}%`);
  card.style.setProperty("--zoom-y", `${originY.toFixed(2)}%`);
  card.classList.toggle("is-zoomed", scale > 1.01);
}

function resetImageZoom(root = document) {
  root.querySelectorAll(".example-image-card").forEach((card) => {
    card.classList.remove("is-zoomed");
    card.classList.remove("is-dragging");
    delete card.dataset.zoomScale;
    delete card.dataset.zoomPanX;
    delete card.dataset.zoomPanY;
    delete card.dataset.zoomOriginX;
    delete card.dataset.zoomOriginY;
    card.style.removeProperty("--zoom-scale");
    card.style.removeProperty("--zoom-pan-x");
    card.style.removeProperty("--zoom-pan-y");
    card.style.removeProperty("--zoom-x");
    card.style.removeProperty("--zoom-y");
  });
}

function captureZoomState(root) {
  return Array.from(root.querySelectorAll(".example-image-card")).map((card) => getCardZoom(card));
}

function applyZoomState(root, zoomState) {
  if (!Array.isArray(zoomState)) return;
  root.querySelectorAll(".example-image-card").forEach((card, index) => {
    if (zoomState[index]) setCardZoom(card, zoomState[index]);
  });
}

function zoomCardAt(card, event, factor) {
  const rect = card.getBoundingClientRect();
  const originX = ((event.clientX - rect.left) / rect.width) * 100;
  const originY = ((event.clientY - rect.top) / rect.height) * 100;
  const zoom = getCardZoom(card);
  const nextScale = clamp(zoom.scale * factor, 1, MAX_ZOOM);

  setCardZoom(card, {
    ...zoom,
    scale: nextScale,
    originX,
    originY,
    panX: nextScale <= 1.01 ? 0 : zoom.panX,
    panY: nextScale <= 1.01 ? 0 : zoom.panY,
  });
}

function bindZoomControls(root, isEnabled) {
  root.querySelectorAll(".example-image-card").forEach((card) => {
    let startX = 0;
    let startY = 0;
    let startPanX = 0;
    let startPanY = 0;
    let moved = false;

    card.addEventListener("pointerdown", (event) => {
      if (!isEnabled()) return;
      const zoom = getCardZoom(card);
      if (zoom.scale <= 1.01) return;

      startX = event.clientX;
      startY = event.clientY;
      startPanX = zoom.panX;
      startPanY = zoom.panY;
      moved = false;
      card.classList.add("is-dragging");
      card.setPointerCapture(event.pointerId);
      event.preventDefault();
    });

    card.addEventListener("pointermove", (event) => {
      if (!card.classList.contains("is-dragging")) return;
      const dx = event.clientX - startX;
      const dy = event.clientY - startY;
      if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
      const zoom = getCardZoom(card);
      setCardZoom(card, {
        ...zoom,
        panX: startPanX + dx,
        panY: startPanY + dy,
      });
    });

    const stopDrag = (event) => {
      if (!card.classList.contains("is-dragging")) return;
      card.classList.remove("is-dragging");
      if (card.hasPointerCapture(event.pointerId)) card.releasePointerCapture(event.pointerId);
      if (moved) {
        card.dataset.suppressNextClick = "true";
        window.setTimeout(() => {
          delete card.dataset.suppressNextClick;
        }, 0);
      }
    };

    card.addEventListener("pointerup", stopDrag);
    card.addEventListener("pointercancel", stopDrag);

    card.addEventListener("click", (event) => {
      if (!isEnabled()) return;
      if (card.dataset.suppressNextClick === "true") return;
      zoomCardAt(card, event, ZOOM_STEP);
    });

    card.addEventListener("wheel", (event) => {
      if (!isEnabled()) return;
      event.preventDefault();
      zoomCardAt(card, event, event.deltaY < 0 ? WHEEL_ZOOM_STEP : 1 / WHEEL_ZOOM_STEP);
    }, { passive: false });
  });
}

function setPairMode(
  viewer,
  edited,
  toggleSelector,
  uneditedText = "Show target/retouched image",
  editedText = "Show source/query image"
) {
  viewer.querySelectorAll(".example-image").forEach((image) => {
    image.src = edited ? image.dataset.afterSrc : image.dataset.beforeSrc;
  });

  viewer.querySelectorAll(".image-state-label").forEach((label) => {
    label.textContent = edited ? label.dataset.afterLabel : label.dataset.beforeLabel;
  });

  const toggle = viewer.querySelector(toggleSelector);
  if (toggle) {
    toggle.textContent = edited ? editedText : uneditedText;
    toggle.setAttribute("aria-pressed", edited ? "true" : "false");
    toggle.classList.toggle("is-edited", edited);
  }
}

function setExampleMode(viewer, edited) {
  setPairMode(viewer, edited, ".example-toggle");
}

function setExampleTabActive(subset) {
  document.querySelectorAll("[data-subset-tab]").forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.subsetTab === subset);
  });
}

function setQualitativeTabActive(subset) {
  document.querySelectorAll("[data-qualitative-subset-tab]").forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.qualitativeSubsetTab === subset);
  });
}

function setExpanded(expanded) {
  const stage = document.querySelector(".example-stage");
  const expand = document.querySelector(".example-expand");
  state.expanded = expanded;

  if (stage) stage.classList.toggle("is-expanded", expanded);
  document.body.classList.toggle("example-expanded", expanded);
  if (!expanded && stage) resetImageZoom(stage);

  if (expand) {
    expand.setAttribute("aria-label", expanded ? "Exit enlarged example viewer" : "Enlarge example viewer");
    expand.setAttribute("title", expanded ? "Exit enlarged example viewer" : "Enlarge example viewer");
    expand.innerHTML = `<i class="fas ${expanded ? "fa-compress" : "fa-expand"}"></i>`;
  }
}

function setQualitativeExpanded(expanded) {
  const browser = document.querySelector(".qualitative-browser");
  const expand = document.querySelector(".qualitative-expand");
  qualitativeState.expanded = expanded;

  if (browser) browser.classList.toggle("is-expanded", expanded);
  document.body.classList.toggle("qualitative-expanded", expanded);
  if (!expanded && browser) resetImageZoom(browser);

  if (expand) {
    expand.setAttribute("aria-label", expanded ? "Exit enlarged comparison viewer" : "Enlarge comparison viewer");
    expand.setAttribute("title", expanded ? "Exit enlarged comparison viewer" : "Enlarge comparison viewer");
    expand.innerHTML = `<i class="fas ${expanded ? "fa-compress" : "fa-expand"}"></i>`;
  }
}

function bindExampleControls(viewer) {
  const toggle = viewer.querySelector(".example-toggle");
  if (toggle) {
    toggle.addEventListener("click", () => {
      state.edited = !state.edited;
      setExampleMode(viewer, state.edited);
    });
  }

  const expand = viewer.querySelector(".example-expand");
  if (expand) {
    expand.addEventListener("click", () => {
      setExpanded(!state.expanded);
    });
  }

  bindZoomControls(viewer, () => state.expanded);
}

function renderExample() {
  const viewer = document.getElementById("example-viewer");
  if (!viewer) return;

  const examples = currentExamples();
  if (examples.length === 0) {
    viewer.innerHTML = "<p>No examples available.</p>";
    return;
  }

  state.index = ((state.index % examples.length) + examples.length) % examples.length;
  const example = examples[state.index];
  state.edited = false;

  viewer.innerHTML = `
    <div class="example-instruction">“${escapeHtml(example.operation)}”</div>
    <button class="example-expand" type="button" aria-label="Enlarge example viewer" title="Enlarge example viewer">
      <i class="fas fa-expand"></i>
    </button>
    <div class="example-pairs">
      ${exampleImageCard(
        example.images.exampleBefore,
        example.images.exampleAfter,
        "Source",
        "Target",
        `${example.operation} exemplar`
      )}
      ${exampleImageCard(
        example.images.queryBefore,
        example.images.ours,
        "Query",
        "Retouched",
        `${example.operation} query transfer`
      )}
    </div>
    <div class="example-toggle-row">
      <button class="button example-toggle" type="button" aria-pressed="false">Show target/retouched image</button>
    </div>
  `;

  bindExampleControls(viewer);
  setExpanded(state.expanded);
}

function currentQualitativeCases() {
  return QUALITATIVE_CASES.filter((caseItem) => caseItem.subsetKey === qualitativeState.subset);
}

function currentQualitativeCase() {
  const cases = currentQualitativeCases();
  if (cases.length === 0) return null;
  qualitativeState.index =
    ((qualitativeState.index % cases.length) + cases.length) % cases.length;
  return cases[qualitativeState.index];
}

function currentQualitativeBaseline(caseItem) {
  if (!caseItem || !Array.isArray(caseItem.baselines) || caseItem.baselines.length === 0) return null;
  return (
    caseItem.baselines.find((baseline) => baseline.key === qualitativeState.baseline) ||
    caseItem.baselines[0]
  );
}

function setQualitativeMode(viewer, edited) {
  setPairMode(
    viewer,
    edited,
    ".qualitative-toggle",
    "Show target/retouched image",
    "Show source/query image"
  );
}

function renderQualitative(renderOptions = {}) {
  const viewer = document.getElementById("qualitative-viewer");
  if (!viewer) return;

  const caseItem = currentQualitativeCase();
  if (!caseItem) {
    viewer.innerHTML = "<p>No qualitative examples available.</p>";
    return;
  }

  const baseline = currentQualitativeBaseline(caseItem);
  qualitativeState.baseline = baseline ? baseline.key : "";
  if (!renderOptions.preserveMode) qualitativeState.edited = false;

  const baselineOptions = caseItem.baselines
    .map((item) => `<option value="${escapeHtml(item.key)}"${item.key === qualitativeState.baseline ? " selected" : ""}>${escapeHtml(item.label)}</option>`)
    .join("");

  viewer.innerHTML = `
    <div class="qualitative-toolbar">
      <label class="qualitative-select-label">
        <span>Baseline</span>
        <select class="qualitative-select">${baselineOptions}</select>
      </label>
    </div>
    <div class="qualitative-heading-row">
      <div class="example-instruction qualitative-instruction">“${escapeHtml(caseItem.operation)}”</div>
      <button class="example-expand qualitative-expand" type="button" aria-label="Enlarge comparison viewer" title="Enlarge comparison viewer">
        <i class="fas fa-expand"></i>
      </button>
    </div>
    <div class="qualitative-pairs">
      ${exampleImageCard(
        caseItem.images.exampleBefore,
        caseItem.images.exampleAfter,
        "Source",
        "Target",
        `${caseItem.operation} exemplar`
      )}
      ${exampleImageCard(
        caseItem.images.queryBefore,
        caseItem.images.ours,
        "Query",
        caseItem.model,
        `${caseItem.operation} MirrorPPR result`
      )}
      ${baseline ? exampleImageCard(
        caseItem.images.queryBefore,
        baseline.image,
        "Query",
        baseline.label,
        `${caseItem.operation} ${baseline.label} result`
      ) : exampleImageCard(
        caseItem.images.queryBefore,
        caseItem.images.ours,
        "Query",
        caseItem.model,
        `${caseItem.operation} MirrorPPR result`
      )}
    </div>
    <div class="example-toggle-row">
      <button class="button example-toggle qualitative-toggle" type="button" aria-pressed="false">Show target/retouched image</button>
    </div>
  `;

  const select = viewer.querySelector(".qualitative-select");
  if (select) {
    select.addEventListener("change", () => {
      const zoomState = captureZoomState(viewer);
      qualitativeState.baseline = select.value;
      renderQualitative({ preserveMode: true, zoomState });
    });
  }

  const toggle = viewer.querySelector(".qualitative-toggle");
  if (toggle) {
    toggle.addEventListener("click", () => {
      qualitativeState.edited = !qualitativeState.edited;
      setQualitativeMode(viewer, qualitativeState.edited);
    });
  }

  const expand = viewer.querySelector(".qualitative-expand");
  if (expand) {
    expand.addEventListener("click", () => {
      setQualitativeExpanded(!qualitativeState.expanded);
    });
  }

  bindZoomControls(viewer, () => qualitativeState.expanded);
  setQualitativeExpanded(qualitativeState.expanded);
  setQualitativeMode(viewer, qualitativeState.edited);
  applyZoomState(viewer, renderOptions.zoomState);
}

function setupExampleControls() {
  document.querySelectorAll("[data-subset-tab]").forEach((tab) => {
    tab.addEventListener("click", () => {
      state.subset = tab.dataset.subsetTab;
      state.index = 0;
      state.edited = false;
      setExampleTabActive(state.subset);
      renderExample();
    });
  });

  const prev = document.getElementById("example-prev");
  const next = document.getElementById("example-next");

  if (prev) {
    prev.addEventListener("click", () => {
      state.index -= 1;
      renderExample();
    });
  }

  if (next) {
    next.addEventListener("click", () => {
      state.index += 1;
      renderExample();
    });
  }
}

function setupQualitativeControls() {
  document.querySelectorAll("[data-qualitative-subset-tab]").forEach((tab) => {
    tab.addEventListener("click", () => {
      qualitativeState.subset = tab.dataset.qualitativeSubsetTab;
      qualitativeState.index = 0;
      qualitativeState.edited = false;
      qualitativeState.baseline = "";
      setQualitativeTabActive(qualitativeState.subset);
      renderQualitative();
    });
  });

  const prev = document.getElementById("qualitative-prev");
  const next = document.getElementById("qualitative-next");

  if (prev) {
    prev.addEventListener("click", () => {
      qualitativeState.index -= 1;
      renderQualitative();
    });
  }

  if (next) {
    next.addEventListener("click", () => {
      qualitativeState.index += 1;
      renderQualitative();
    });
  }
}

function setupCopyBibTeX() {
  const button = document.querySelector(".copy-bibtex-btn");
  const code = document.getElementById("bibtex-code");
  if (!button || !code) return;

  button.addEventListener("click", async () => {
    const text = code.textContent.trim();

    try {
      await navigator.clipboard.writeText(text);
    } catch (error) {
      const textArea = document.createElement("textarea");
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      document.body.removeChild(textArea);
    }

    button.textContent = "Copied";
    button.classList.add("copied");
    setTimeout(() => {
      button.textContent = "Copy";
      button.classList.remove("copied");
    }, 200);
  });
}

function setupKeyboardControls() {
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;

    if (state.expanded) {
      setExpanded(false);
    }

    if (qualitativeState.expanded) {
      setQualitativeExpanded(false);
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setupExampleControls();
  setupQualitativeControls();
  setupCopyBibTeX();
  setupKeyboardControls();
  setExampleTabActive(state.subset);
  setQualitativeTabActive(qualitativeState.subset);
  renderExample();
  renderQualitative();
});
