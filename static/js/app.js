document.querySelectorAll(".status").forEach(s=>s.addEventListener("change",async()=>{const t=document.getElementById("toast");try{const r=await fetch(`/api/alerts/${s.dataset.id}/status`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({status:s.value})});const d=await r.json();if(!r.ok||!d.ok)throw Error(d.error||"Update failed");t.textContent=`Alert #${s.dataset.id} updated to ${s.value}.`;t.className="show";setTimeout(()=>t.className="",2500)}catch(e){t.textContent=e.message;t.className="show errorToast";setTimeout(()=>t.className="",3500)}}));
/* BankSphere Neural Orbit Cursor V2: responsive, low-cost motion */
(() => {
  const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  if (!finePointer.matches || reducedMotion.matches) return;
  const cursor = document.querySelector('.neural-cursor');
  const rippleLayer = document.querySelector('.cursor-ripple-layer');
  if (!cursor || !rippleLayer) return;
  document.documentElement.classList.add('banksphere-custom-cursor');

  let targetX = -100, targetY = -100, ringX = targetX, ringY = targetY;
  let rafId = 0, lastSparkAt = 0, lastSparkX = targetX, lastSparkY = targetY;
  const ease = 0.34;
  function animate() {
    ringX += (targetX - ringX) * ease;
    ringY += (targetY - ringY) * ease;
    cursor.style.transform = `translate3d(${ringX}px, ${ringY}px, 0)`;
    if (Math.abs(targetX-ringX) > .35 || Math.abs(targetY-ringY) > .35) {
      rafId = requestAnimationFrame(animate);
    } else { rafId = 0; }
  }
  window.addEventListener('mousemove', (event) => {
    targetX = event.clientX; targetY = event.clientY;
    cursor.classList.add('is-visible');
    if (!rafId) rafId = requestAnimationFrame(animate);
    const now = performance.now(), dx = targetX-lastSparkX, dy = targetY-lastSparkY;
    if (now-lastSparkAt > 160 && (dx*dx+dy*dy) > 1600) {
      const spark = document.createElement('span');
      spark.className = 'cursor-spark';
      spark.style.left = `${targetX}px`; spark.style.top = `${targetY}px`;
      spark.style.setProperty('--spark-x', `${(Math.random()-.5)*20}px`);
      spark.style.setProperty('--spark-y', `${(Math.random()-.5)*20}px`);
      document.body.appendChild(spark);
      window.setTimeout(() => spark.remove(), 460);
      lastSparkAt = now; lastSparkX = targetX; lastSparkY = targetY;
    }
  }, { passive: true });
  document.addEventListener('mouseover', (event) => {
    const target = event.target instanceof Element ? event.target : null;
    cursor.classList.toggle('is-hovering', !!target?.closest('a, button, [role="button"], input, select, textarea, .clickable'));
  }, { passive: true });
  document.addEventListener('mousedown', (event) => {
    cursor.classList.add('is-clicking');
    const ripple = document.createElement('span');
    ripple.className = 'cursor-ripple';
    ripple.style.left = `${event.clientX}px`; ripple.style.top = `${event.clientY}px`;
    rippleLayer.appendChild(ripple);
    window.setTimeout(() => ripple.remove(), 450);
  }, { passive: true });
  document.addEventListener('mouseup', () => cursor.classList.remove('is-clicking'), { passive: true });
  document.addEventListener('mouseleave', () => cursor.classList.remove('is-visible'));
  document.addEventListener('mouseenter', () => cursor.classList.add('is-visible'));
})();


/* BOB message cycle: sequence and 30-second schedule persist across page navigation */
(() => {
  const box = document.querySelector('.bob-companion');
  const messageEl = document.getElementById('bob-message');
  if (!box || !messageEl) return;

  const messages = [
    'Hey, amigo! BOB here.',
    'You can call me BOB.',
    "By the way, how’s it going?",
    'Am I annoying you? 😅',
    'Hey… wanna hear some nonsense?',
    'BOB says: “Chill, it’s all good.” 😎',
    'Adiós, amigo… see you around!'
  ];
  const STORAGE_KEY = 'banksphere_bob_state_v1';
  const INTERVAL = 10000;
  const VISIBLE_MS = 6500;
  let dismissed = false;
  let visibleTimer = null;
  let cycleTimer = null;

  function readState() {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (raw) {
        const state = JSON.parse(raw);
        if (Number.isInteger(state.index) && typeof state.nextAt === 'number') return state;
      }
    } catch (_) {}
    const state = { index: 0, nextAt: Date.now() + INTERVAL };
    saveState(state);
    return state;
  }
  function saveState(state) {
    try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch (_) {}
  }
  function showCurrent() {
    if (dismissed) return;
    const state = readState();
    messageEl.classList.remove('is-changing');
    void messageEl.offsetWidth;
    messageEl.textContent = messages[state.index];
    messageEl.classList.add('is-changing');
    box.classList.add('is-visible');
    clearTimeout(visibleTimer);
    visibleTimer = setTimeout(() => box.classList.remove('is-visible'), VISIBLE_MS);
  }
  function tick() {
    if (dismissed) return;
    const state = readState();
    const now = Date.now();
    if (now >= state.nextAt) {
      const elapsedSlots = Math.max(1, Math.floor((now - state.nextAt) / INTERVAL) + 1);
      state.index = (state.index + elapsedSlots) % messages.length;
      state.nextAt += elapsedSlots * INTERVAL;
      saveState(state);
      showCurrent();
    }
    cycleTimer = setTimeout(tick, Math.max(250, Math.min(1000, state.nextAt - Date.now())));
  }
  // Keep timer accurate when returning from another page or inactive tab.
  const state = readState();
  if (Date.now() >= state.nextAt) {
    const elapsedSlots = Math.floor((Date.now() - state.nextAt) / INTERVAL) + 1;
    state.index = (state.index + elapsedSlots) % messages.length;
    state.nextAt += elapsedSlots * INTERVAL;
    saveState(state);
  }
  showCurrent();
  tick();
  window.addEventListener('pageshow', () => { if (!dismissed) { showCurrent(); } });
  document.addEventListener('visibilitychange', () => { if (!document.hidden && !dismissed) tick(); });

  const close = box.querySelector('.bob-dismiss');
  if (close) close.addEventListener('click', () => {
    dismissed = true;
    box.classList.remove('is-visible');
    clearTimeout(visibleTimer);
    clearTimeout(cycleTimer);
  });
})();
