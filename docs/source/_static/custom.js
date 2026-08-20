// Collapsible sidebar captions. Furo only makes a page collapsible when the
// toctree nests it under another page; this site's toctree is flat, so no
// caption ("BP-FORMAT GUIDE", "GALLERY", ...) is ever collapsible on its own.
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
});
