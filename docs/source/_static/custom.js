// Collapsible sidebar captions. Furo only makes a page collapsible when the
// toctree nests it under another page; this site's toctree is flat, so no
// caption ("HYBRAX.FORMAT GUIDE", "GALLERY", ...) is ever collapsible on its own.
// This makes each caption itself the toggle for the page list under it,
// expanded by default for "Start here" and "Tutorials" plus whichever
// section holds the current page; everything else starts collapsed.
var ALWAYS_OPEN_CAPTIONS = ["Start here", "Tutorials"];

document.addEventListener("DOMContentLoaded", function () {
  var captions = document.querySelectorAll(".sidebar-tree p.caption");

  captions.forEach(function (caption) {
    var list = caption.nextElementSibling;
    if (!list || list.tagName !== "UL") {
      return;
    }

    var icon = document.createElement("span");
    icon.className = "icon caption-toggle-icon";
    icon.innerHTML = '<svg><use href="#svg-arrow-right"></use></svg>';
    caption.appendChild(icon);

    caption.setAttribute("role", "button");
    caption.setAttribute("tabindex", "0");

    var setExpanded = function (expanded) {
      caption.classList.toggle("collapsed", !expanded);
      caption.setAttribute("aria-expanded", expanded ? "true" : "false");
    };

    var captionText = caption.querySelector(".caption-text").textContent.trim();
    var opensByDefault =
      list.classList.contains("current") ||
      ALWAYS_OPEN_CAPTIONS.indexOf(captionText) !== -1;
    setExpanded(opensByDefault);

    caption.addEventListener("click", function () {
      setExpanded(caption.classList.contains("collapsed"));
    });
    caption.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        setExpanded(caption.classList.contains("collapsed"));
      }
    });
  });

  // --- keep the sidebar's scroll position across page loads --------------
  // Every click loads a fresh document, so the browser has no notion of
  // "the sidebar" surviving navigation: scroll down, click a page, and it
  // snaps back to the top. These docs also work opened straight from
  // file://, where storage APIs are unreliable, so the scroll offset rides
  // along as a URL parameter instead: stamped onto a link's href right
  // before it is followed, then read back and stripped on the page it
  // lands on.
  var scroller = document.querySelector(".sidebar-scroll");
  if (scroller) {
    var params = new URLSearchParams(window.location.search);
    var savedScroll = params.get("navscroll");
    if (savedScroll !== null) {
      scroller.scrollTop = parseInt(savedScroll, 10) || 0;
      // Cosmetic only (tidies the address bar): file:// gives browsers
      // inconsistent, sometimes opaque origins, and replaceState can throw
      // there. Restoring the scroll position above must not be undone by a
      // failure here.
      try {
        params.delete("navscroll");
        var query = params.toString();
        window.history.replaceState(
          null, "",
          window.location.pathname + (query ? "?" + query : "") + window.location.hash
        );
      } catch (e) {
        // Leave the URL as-is.
      }
    }

    document.querySelectorAll(".sidebar-tree a.reference.internal[href]")
      .forEach(function (link) {
        link.addEventListener("click", function () {
          try {
            var url = new URL(link.getAttribute("href"), window.location.href);
            url.searchParams.set("navscroll", Math.round(scroller.scrollTop));
            link.href = url.href;
          } catch (e) {
            // Malformed href: leave the link alone.
          }
        });
      });
  }
});
