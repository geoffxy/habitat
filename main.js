document.addEventListener("DOMContentLoaded", function() {
  const BIBTEX_CODE_ID = "bibtex-code";
  const BIBTEX_CONTENT_SELECTOR = ".pub-bibtex";
  const BIBTEX_COPY_SELECTOR = ".pub-bibtex-copy";
  const BIBTEX_TOGGLE_SELECTOR = ".pub-bibtex-toggle";
  const BIBTEX_CODE_SELECTOR = BIBTEX_CONTENT_SELECTOR + " pre";
  const COPIED_TEXT_DELAY_MS = 2000;

  function bindClickHandler(selector, fn) {
    for (const el of document.querySelectorAll(selector)) {
      el.addEventListener("click", fn);
    }
  }

  function toggleDisplayStyle(el) {
    const current = el.style.display;
    if (current === "" || current === "none") {
      el.style.display = "block";
    } else {
      el.style.display = "none";
    }
    return current === "block";
  }

  bindClickHandler(BIBTEX_TOGGLE_SELECTOR, function(event) {
    event.preventDefault();
    const codeEl = document.getElementById(BIBTEX_CODE_ID);
    const wasVisible = toggleDisplayStyle(codeEl);
    if (wasVisible) {
      event.target.textContent = "Show BibTeX";
    } else {
      event.target.textContent = "Hide BibTeX";
    }
  });

  bindClickHandler(BIBTEX_COPY_SELECTOR, function(event) {
    event.preventDefault();
    const entryEl = event.target.closest(BIBTEX_CONTENT_SELECTOR);
    const bibtexCodeEl = entryEl.querySelector(BIBTEX_CODE_SELECTOR);
    navigator.clipboard.writeText(bibtexCodeEl.textContent);
    event.target.textContent = "Copied";
    setTimeout(function() {
      event.target.textContent = "Copy";
    }, COPIED_TEXT_DELAY_MS);
  });
});
