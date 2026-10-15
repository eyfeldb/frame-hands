/* Native Steam UI adapter for changelist 11094443. Loaded in the bar's CDP context.
 * Reuses Steam's React, navigation providers, tooltip and small-button component.
 * No Steam asset file is modified. The supervisor verifies hashes before loading.
 */
(() => {
  const VERSION = 'frame-hands-dashboard-v1';
  if (window.__frameHandsDashboard?.version === VERSION) {
    window.__frameHandsDashboard.ensureMounted();
    return window.__frameHandsDashboard.info();
  }
  const owner = window.opener;
  if (!owner?.webpackChunksteamui) throw Error('Native Steam UI owner unavailable');
  let require = owner.__frameHandsWebpackRequire;
  if (!require) {
    owner.webpackChunksteamui.push([[VERSION + ':' + Date.now()], {}, r => { require = r; }]);
    owner.__frameHandsWebpackRequire = require;
  }
  const React = require(63696), DOM = require(78325), Client = require(7470);
  const NativeButton = require(34493).l9;
  if (!NativeButton || !Client.createRoot || !DOM.flushSync) throw Error('Native button API changed');
  const h = React.createElement;
  const glyph = 'M14.5 33C12.8 31.5 11.1 29.5 9.6 27.5L3.9 20.2C3.1 19.1 3.3 17.7 4.3 17C5.3 16.3 6.6 16.6 7.4 17.5L10.1 20.8L9.6 8.3C9.55 7.05 10.4 6.1 11.55 6.05C12.7 6 13.6 6.9 13.65 8.1L14.1 16.8C14.12 17.22 14.43 17.5 14.82 17.5C15.22 17.5 15.5 17.2 15.5 16.8V4.55C15.5 3.35 16.38 2.4 17.55 2.4C18.72 2.4 19.6 3.35 19.6 4.55V16.2C19.6 16.62 19.91 16.9 20.3 16.9C20.69 16.9 21 16.62 21 16.2L21.35 6.5C21.39 5.3 22.3 4.4 23.45 4.45C24.6 4.5 25.44 5.45 25.4 6.65L25.05 18C25.04 18.4 25.33 18.7 25.72 18.72C26.12 18.74 26.42 18.46 26.45 18.05L27 12C27.1 10.9 27.98 10.15 29.05 10.24C30.12 10.33 30.84 11.2 30.75 12.3L29.85 23.6C29.61 27.07 28.32 30.02 26.1 33H14.5Z';
  let state = Object.freeze({enabled: false, busy: false, ready: false, error: ''});
  let root, container, anchor, providerSnapshot = [], lastUpdate = 0;
  const listeners = new Set();
  const subscribe = callback => { listeners.add(callback); return () => listeners.delete(callback); };
  const snapshot = () => state;
  function update(value) {
    lastUpdate = Date.now();
    state = Object.freeze({...state, ...value});
    listeners.forEach(callback => callback());
  }
  function activate() {
    if (!state.ready || state.busy || typeof window.frameHandsRequest !== 'function') return;
    const enabled = !state.enabled;
    update({busy: true, error: ''});
    window.frameHandsRequest(JSON.stringify({enabled}));
  }
  function HandsButton() {
    const status = React.useSyncExternalStore(subscribe, snapshot);
    return h(NativeButton, {
      active: status.enabled, onActivate: activate,
      tooltip: status.error ? 'Hands: ' + status.error : status.busy ? (status.enabled ? 'Stopping hands…' : 'Starting hands…') : 'Hands',
      'aria-label': 'Hands', 'aria-pressed': status.enabled,
      'aria-busy': status.busy, 'aria-disabled': !status.ready || status.busy,
      'data-frame-hands-toggle': VERSION
    }, h('svg', {viewBox: '0 0 36 36', fill: 'none', 'aria-hidden': true},
      h('path', {fill: 'currentColor', d: glyph})));
  }
  function fiberFor(element) {
    const fiber = element[Object.keys(element).find(k => k.startsWith('__reactFiber$'))];
    let top = fiber;
    while (top?.return) top = top.return;
    return top?.stateNode?.current && top.stateNode.current !== top && fiber?.alternate ? fiber.alternate : fiber;
  }
  function findAnchor() {
    for (const element of document.querySelectorAll('.VRDashboardBarSmallButton')) {
      if (element.hasAttribute('data-frame-hands-toggle')) continue;
      let fiber = fiberFor(element);
      for (let n = 0; fiber && n < 15; n++, fiber = fiber.return) {
        const action = fiber.memoizedProps?.action;
        if (action?.invocation === 2 && action.icon?.enum === 2 && action.icon_active?.enum === 1) {
          return element;
        }
      }
    }
    return null;
  }
  function unmount() {
    if (root) DOM.flushSync(() => root.unmount());
    root = undefined;
    container?.remove(); container = undefined; anchor = undefined; providerSnapshot = [];
  }
  function ensureMounted() {
    const nextAnchor = findAnchor();
    if (!nextAnchor) { if (container) unmount(); return false; }
    const group = nextAnchor.parentElement;
    const parentFiber = fiberFor(group);
    const providers = [], seen = new Set();
    for (let fiber = parentFiber; fiber; fiber = fiber.return) {
      if (fiber.tag === 10 && !seen.has(fiber.type)) {
        seen.add(fiber.type); providers.push([fiber.type, fiber.memoizedProps.value]);
      }
    }
    if (providers.length < 5) throw Error('Native navigation context unavailable');
    if (container?.isConnected && anchor === nextAnchor) {
      if (nextAnchor.nextElementSibling !== container) nextAnchor.after(container);
      if (providerSnapshot.length === providers.length && providers.every(([type, value], i) => providerSnapshot[i][0] === type && providerSnapshot[i][1] === value)) return true;
    } else {
      unmount(); anchor = nextAnchor;
      container = document.createElement('div');
      container.id = 'frame-hands-dashboard-slot';
      container.style.display = 'flex';
      nextAnchor.after(container);
      root = Client.createRoot(container);
    }
    let tree = h(HandsButton);
    // Nearest provider wins; wrap from nearest to outermost without changing Valve's tree.
    for (const [type, value] of providers) tree = h(type, {value}, tree);
    DOM.flushSync(() => root.render(tree));
    providerSnapshot = providers;
    return true;
  }
  const timer = setInterval(() => {
    if (state.ready && Date.now() - lastUpdate > 5000) update({ready: false, busy: false, error: 'Connection lost'});
  }, 1000);
  window.__frameHandsDashboard = {
    version: VERSION, update, ensureMounted,
    info: () => ({version: VERSION, mounted: !!container?.querySelector('[data-frame-hands-toggle]'), ...state}),
    dispose: () => { clearInterval(timer); unmount(); delete window.__frameHandsDashboard; }
  };
  try { ensureMounted(); } catch (error) { window.__frameHandsDashboard.dispose(); throw error; }
  return window.__frameHandsDashboard.info();
})()
