"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const sections = document.querySelectorAll("main section[id]");
  const navLinks = document.querySelectorAll(".nav-link");

  const scrollToHash = (hash, updateHistory = true) => {
    if (!hash || hash === "#") return false;

    const target = document.querySelector(hash);
    if (!target) return false;

    target.scrollIntoView({ behavior: "smooth", block: "start" });

    if (updateHistory) {
      history.pushState(null, "", hash);
    }

    return true;
  };

  const setActiveLink = () => {
    let activeId = "home";

    sections.forEach((section) => {
      const offsetTop = section.offsetTop - 120;
      if (window.scrollY >= offsetTop) {
        activeId = section.id;
      }
    });

    navLinks.forEach((link) => {
      const linkHash = new URL(link.href, window.location.href).hash;
      link.classList.toggle("active", linkHash === `#${activeId}`);
    });
  };

  document.querySelectorAll('a[href*="#"]').forEach((link) => {
    link.addEventListener("click", (event) => {
      const url = new URL(link.href, window.location.href);
      const samePage =
        url.origin === window.location.origin &&
        url.pathname === window.location.pathname;

      if (!samePage) return;

      if (scrollToHash(url.hash)) {
        event.preventDefault();
        setActiveLink();
      }
    });
  });

  if (window.location.hash) {
    window.setTimeout(() => {
      scrollToHash(window.location.hash, false);
      setActiveLink();
    }, 0);
  }

  setActiveLink();
  window.addEventListener("scroll", setActiveLink, { passive: true });
});
