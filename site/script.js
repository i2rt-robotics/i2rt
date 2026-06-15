/* I2RT docs — sidebar active state, TOC scrollspy, search filter, theme, mobile nav */
(function () {
  "use strict";

  // ---- Theme ----
  const root = document.documentElement;
  const saved = localStorage.getItem("i2rt-theme");
  if (saved) root.setAttribute("data-theme", saved);
  function toggleTheme() {
    const cur = root.getAttribute("data-theme") === "light" ? "dark" : "light";
    root.setAttribute("data-theme", cur);
    localStorage.setItem("i2rt-theme", cur);
    updateThemeLabel();
  }
  function updateThemeLabel() {
    const isLight = root.getAttribute("data-theme") === "light";
    document.querySelectorAll(".theme-toggle").forEach(b => {
      b.textContent = isLight ? "🌙  Dark mode" : "☀️  Light mode";
    });
  }
  document.querySelectorAll(".theme-toggle").forEach(b => b.addEventListener("click", toggleTheme));
  updateThemeLabel();

  // ---- Mobile sidebar ----
  const sidebar = document.querySelector(".sidebar");
  const scrim = document.querySelector(".scrim");
  const burger = document.querySelector("#burger");
  function closeNav() { sidebar.classList.remove("open"); if (scrim) scrim.classList.remove("show"); }
  if (burger) burger.addEventListener("click", () => {
    sidebar.classList.toggle("open");
    if (scrim) scrim.classList.toggle("show");
  });
  if (scrim) scrim.addEventListener("click", closeNav);

  // ---- Build TOC from h2/h3 ----
  const main = document.querySelector("main");
  const tocList = document.querySelector("#toc-list");
  const headings = [...main.querySelectorAll("h2[id], h3[id]")];
  if (tocList) {
    headings.forEach(h => {
      const a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.textContent;
      if (h.tagName === "H3") a.classList.add("lvl3");
      tocList.appendChild(a);
    });
  }

  // ---- Scrollspy (sidebar sections + TOC) ----
  const navLinks = [...document.querySelectorAll(".sidebar a[href^='#']")];
  const tocLinks = [...document.querySelectorAll("#toc-list a")];
  const sections = [...main.querySelectorAll("section[id]")];
  const allHeads = [...main.querySelectorAll("h2[id], h3[id]")];

  function spy() {
    const y = window.scrollY + 100;
    // sidebar: by section
    let curSec = sections[0] && sections[0].id;
    for (const s of sections) { if (s.offsetTop <= y) curSec = s.id; }
    navLinks.forEach(a => a.classList.toggle("active", a.getAttribute("href") === "#" + curSec));
    // toc: by heading
    let curHead = allHeads[0] && allHeads[0].id;
    for (const h of allHeads) { if (h.offsetTop <= y) curHead = h.id; }
    tocLinks.forEach(a => a.classList.toggle("active", a.getAttribute("href") === "#" + curHead));
  }
  window.addEventListener("scroll", spy, { passive: true });
  window.addEventListener("resize", spy);
  spy();

  navLinks.forEach(a => a.addEventListener("click", () => { if (window.innerWidth <= 900) closeNav(); }));

  // ---- Sidebar search filter ----
  const search = document.querySelector("#nav-search-input");
  if (search) {
    search.addEventListener("input", () => {
      const q = search.value.trim().toLowerCase();
      document.querySelectorAll(".nav-group").forEach(group => {
        let anyVisible = false;
        group.querySelectorAll("a").forEach(a => {
          const match = a.textContent.toLowerCase().includes(q);
          a.style.display = match ? "" : "none";
          if (match) anyVisible = true;
        });
        const title = group.querySelector(".nav-title");
        if (title) title.style.display = anyVisible ? "" : "none";
      });
    });
  }

  // ---- Copy buttons on code blocks ----
  document.querySelectorAll("pre").forEach(pre => {
    const btn = document.createElement("button");
    btn.textContent = "Copy";
    Object.assign(btn.style, {
      position: "absolute", top: "8px", right: "8px",
      fontSize: "11px", padding: "3px 9px", cursor: "pointer",
      background: "var(--bg-elevated)", color: "var(--text-soft)",
      border: "1px solid var(--border)", borderRadius: "6px", opacity: "0",
      transition: "opacity .15s"
    });
    pre.addEventListener("mouseenter", () => (btn.style.opacity = "1"));
    pre.addEventListener("mouseleave", () => (btn.style.opacity = "0"));
    btn.addEventListener("click", () => {
      const code = pre.querySelector("code");
      navigator.clipboard.writeText(code ? code.innerText : pre.innerText);
      btn.textContent = "Copied!";
      setTimeout(() => (btn.textContent = "Copy"), 1400);
    });
    pre.appendChild(btn);
  });
})();
