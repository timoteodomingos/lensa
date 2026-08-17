window.addEventListener('load', () => {
  const tip = document.querySelector('.deck-tooltip');
  const cmap = document.getElementById('colormap-selector-container');

  if (!tip || !cmap) {
    console.warn('[mobile.js] missing element', { tip: !!tip, cmap: !!cmap });
    return;
  }

  const sync = () => {
    const showing =
      window.innerWidth <= 768 &&
      tip.style.display !== 'none' &&
      tip.innerHTML.trim() !== '';
    cmap.style.visibility = showing ? 'hidden' : '';
  };

  new MutationObserver(sync).observe(tip, {
    attributes: true,
    childList: true,
    subtree: true,
    attributeFilter: ['style'],
  });

  window.addEventListener('resize', sync);
  sync();

  window.__syncInstalled = true;
  console.log('[mobile.js] installed');
});
