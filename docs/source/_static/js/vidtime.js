/* Client side of the :vidtime: role (docs/source/_ext/vidtime.py).
   A link carrying data-vidtime-target/-seconds seeks the <video> inside that figure, scrolls it
   into view and focuses it. The same works as a deep link: page.html#figure-id@83 (or @1:23). */
(function () {
  "use strict";

  function parseTime(text) {
    var parts = String(text).split(":"), s = 0, i;
    if (parts.length > 3) return NaN;
    for (i = 0; i < parts.length; i++) {
      var p = parseFloat(parts[i]);
      if (isNaN(p)) return NaN;
      s = s * 60 + p;
    }
    return s;
  }

  function findVideo(id) {
    var el = document.getElementById(id);
    if (!el) return null;
    return el.tagName === "VIDEO" ? el : el.querySelector("video");
  }

  function seek(id, seconds) {
    var video = findVideo(id);
    if (!video) return false;
    var apply = function () {
      // Videos are muted and looping, so seeking past the end would be silently wrapped.
      var d = video.duration;
      video.currentTime = (d && isFinite(d)) ? Math.min(seconds, Math.max(0, d - 0.05)) : seconds;
      var p = video.play();
      if (p && p.catch) p.catch(function () {});  // autoplay policies: a still frame is fine too
    };
    if (video.readyState >= 1) apply();
    else video.addEventListener("loadedmetadata", apply, { once: true });

    var figure = video.closest("figure") || video;
    figure.scrollIntoView({ behavior: "smooth", block: "center" });
    if (!video.hasAttribute("tabindex")) video.setAttribute("tabindex", "-1");
    video.focus({ preventScroll: true });
    figure.classList.remove("vidtime-flash");
    void figure.offsetWidth;                       // restart the CSS animation on a repeat click
    figure.classList.add("vidtime-flash");
    return true;
  }

  document.addEventListener("click", function (ev) {
    var a = ev.target.closest ? ev.target.closest("a[data-vidtime-target]") : null;
    if (!a) return;
    var id = a.getAttribute("data-vidtime-target");
    var s = parseFloat(a.getAttribute("data-vidtime-seconds"));
    if (seek(id, isNaN(s) ? 0 : s)) {
      ev.preventDefault();
      if (window.history && history.replaceState) history.replaceState(null, "", "#" + id);
    }
  });

  function fromHash() {
    var h = decodeURIComponent(window.location.hash.replace(/^#/, ""));
    var at = h.lastIndexOf("@");
    if (at <= 0) return;
    var t = parseTime(h.slice(at + 1));
    if (!isNaN(t)) seek(h.slice(0, at), t);
  }
  window.addEventListener("hashchange", fromHash);
  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", fromHash);
  else fromHash();
})();
