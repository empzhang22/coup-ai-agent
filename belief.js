// belief.js
// Particle filter belief tracking for hidden cards in Coup.
// Defines global: ParticleBeliefTracker
//
// Expected global: CHARACTERS (array of character names) defined in index.html.
//
// Assumptions (matching your engine):
// - player.id === index in gameState.players[]
// - gameHistory.revealedCards: array per-player of revealed card names
// - gameHistory.actionHistory entries: { playerId, action, targetId?, challenged?, success? }
//   where success=true on challenged claim means claimant HAD the required card.

class ParticleBeliefTracker {
  constructor(selfId, opts = {}) {
    this.selfId = selfId;

    this.N = opts.numParticles ?? 700;
    this.resampleEssFrac = opts.resampleEssFrac ?? 0.55;

    // Soft-evidence model for unchallenged claims
    this.truthLikelihood = opts.truthLikelihood ?? 0.88;
    this.bluffLikelihood = opts.bluffLikelihood ?? 0.12;

    this.repairAttempts = opts.repairAttempts ?? 14;
    this.rng = opts.rng ?? Math.random;

    this.particles = null; // [{ hands: string[][], deck: string[], w: number }]
    this.lastHistoryLen = 0;
    this._lastRevealedKey = "";
  }

  // ------------ Public API ------------

  sync(gameState, gameHistory) {
    const histLen = (gameHistory?.actionHistory?.length ?? 0);
    const revKey = this._fingerprintRevealed(gameHistory);

    const needInit =
      !this.particles ||
      this.particles.length !== this.N ||
      histLen < this.lastHistoryLen ||
      revKey !== this._lastRevealedKey;

    if (needInit) {
      this._initFromPublic(gameState, gameHistory);
      this.lastHistoryLen = histLen;
      this._lastRevealedKey = revKey;
      return;
    }

    for (let i = this.lastHistoryLen; i < histLen; i++) {
      this._applyHistoryEvent(gameState, gameHistory, gameHistory.actionHistory[i]);
    }
    this.lastHistoryLen = histLen;

    const ess = this._effectiveSampleSize();
    if (ess < this.resampleEssFrac * this.N) {
      this._resampleSystematic();
      this._rejuvenate(0.15);
      this._normalizeWeights();
    }
  }

  probHas(playerIdx, card) {
    if (!this.particles || this.particles.length === 0) return 0.2;
    let num = 0, den = 0;
    for (const p of this.particles) {
      den += p.w;
      if ((p.hands[playerIdx] || []).includes(card)) num += p.w;
    }
    return den > 0 ? num / den : 0.2;
  }

  // sample a full hidden world
  sampleParticleWorld() {
    if (!this.particles || this.particles.length === 0) return null;
    const total = this._weightSum();
    const r = this.rng() * (total > 0 ? total : 1);
    let acc = 0;
    for (const p of this.particles) {
      acc += p.w;
      if (acc >= r) return { hands: this._deepCopyHands(p.hands), deck: p.deck.slice() };
    }
    const last = this.particles[this.particles.length - 1];
    return { hands: this._deepCopyHands(last.hands), deck: last.deck.slice() };
  }

  // ------------ Init ------------

  _initFromPublic(gameState, gameHistory) {
    const playerCount = gameState.players.length;

    // Build multiset: 3 copies of each character
    const all = [];
    for (const c of CHARACTERS) for (let i = 0; i < 3; i++) all.push(c);

    // Remove publicly revealed cards
    const revealedCards = gameHistory?.revealedCards;
    if (Array.isArray(revealedCards)) {
      for (const arr of revealedCards) {
        if (!Array.isArray(arr)) continue;
        for (const rc of arr) {
          const idx = all.indexOf(rc);
          if (idx !== -1) all.splice(idx, 1);
        }
      }
    }

    // Remove our own known unrevealed cards
    const self = gameState.players[this.selfId];
    if (self) {
      for (const cardObj of (self.cards || [])) {
        if (!cardObj.revealed) {
          const idx = all.indexOf(cardObj.character);
          if (idx !== -1) all.splice(idx, 1);
        }
      }
    }

    // Hidden card counts for each opponent (alive influences)
    const hiddenCounts = new Array(playerCount).fill(0);
    for (let i = 0; i < playerCount; i++) {
      const pl = gameState.players[i];
      if (!pl || pl.eliminated) continue;
      const alive = (pl.cards || []).filter(c => !c.revealed).length;
      hiddenCounts[i] = (i === this.selfId) ? 0 : alive;
    }

    this.particles = [];
    for (let n = 0; n < this.N; n++) {
      const deck = all.slice();
      this._shuffle(deck);

      const hands = new Array(playerCount);
      for (let i = 0; i < playerCount; i++) hands[i] = [];

      for (let i = 0; i < playerCount; i++) {
        const k = hiddenCounts[i];
        for (let t = 0; t < k; t++) {
          if (deck.length === 0) break;
          hands[i].push(deck.pop());
        }
      }

      this.particles.push({ hands, deck, w: 1.0 });
    }

    this._normalizeWeights();
  }

  // ------------ Filtering ------------

  _applyHistoryEvent(gameState, gameHistory, ev) {
    if (!ev || !ev.action) return;

    const claimantIdx = this._coerceToIndex(gameState, ev.playerId);
    if (claimantIdx < 0) return;

    const required = this._requiredCardForAction(ev.action);
    const isClaim = !!required;

    // Soft evidence: unchallenged claims
    if (isClaim && !ev.challenged) {
      for (const p of this.particles) {
        const has = (p.hands[claimantIdx] || []).includes(required);
        p.w *= has ? this.truthLikelihood : this.bluffLikelihood;
      }
      this._normalizeIfTiny();
      this._normalizeWeights();
      return;
    }

    // Hard evidence: challenged claims
    if (isClaim && ev.challenged) {
      const claimantHad = !!ev.success;

      for (const p of this.particles) {
        if (p.w <= 0) continue;
        const hand = p.hands[claimantIdx] || [];
        const has = hand.includes(required);

        if (claimantHad && !has) {
          if (!this._repairForceHas(p, claimantIdx, required)) p.w = 0;
        } else if (!claimantHad && has) {
          if (!this._repairForceLacks(p, claimantIdx, required)) p.w = 0;
        } else if (claimantHad) {
          // If they had it, model reveal+replace (approximate)
          this._transitionRevealReplace(p, claimantIdx, required);
        }
      }

      // If belief collapses, restart from public
      if (this._weightSum() <= 0) {
        this._initFromPublic(gameState, gameHistory);
      } else {
        this._normalizeWeights();
      }
      return;
    }

    // Exchange transition: increase uncertainty for claimant
    if (ev.action === "exchange") {
      for (const p of this.particles) {
        if (p.w <= 0) continue;
        this._transitionExchange(p, claimantIdx);
      }
      this._normalizeWeights();
    }
  }

  // ------------ Transitions ------------

  _transitionRevealReplace(p, idx, card) {
    const hand = p.hands[idx] || [];
    const pos = hand.indexOf(card);
    if (pos === -1 || p.deck.length === 0) return;

    const di = Math.floor(this.rng() * p.deck.length);
    const newCard = p.deck.splice(di, 1)[0];

    p.deck.push(card);
    hand[pos] = newCard;

    if (this.rng() < 0.35) this._shuffle(p.deck);
  }

  _transitionExchange(p, idx) {
    const hand = p.hands[idx] || [];
    if (hand.length === 0 || p.deck.length === 0) return;

    const draw = Math.min(2, p.deck.length);
    const drawn = [];
    for (let i = 0; i < draw; i++) drawn.push(p.deck.pop());

    const pool = hand.concat(drawn);
    this._shuffle(pool);

    const keepN = hand.length;
    p.hands[idx] = pool.slice(0, keepN);

    const returned = pool.slice(keepN);
    p.deck.push(...returned);
    if (this.rng() < 0.6) this._shuffle(p.deck);
  }

  // ------------ Repairs ------------

  _repairForceHas(p, idx, card) {
    for (let t = 0; t < this.repairAttempts; t++) {
      // from deck?
      const di = p.deck.indexOf(card);
      if (di !== -1) {
        const hand = p.hands[idx] || [];
        if (hand.length === 0) return false;

        const swapPos = Math.floor(this.rng() * hand.length);
        const out = hand[swapPos];
        hand[swapPos] = card;

        p.deck.splice(di, 1);
        p.deck.push(out);
        return true;
      }

      // from someone else's hand?
      const other = this._randomOtherIndex(p.hands.length, idx);
      if (other === -1) return false;
      const ohand = p.hands[other] || [];
      const oi = ohand.indexOf(card);
      if (oi === -1) continue;

      const hand = p.hands[idx] || [];
      if (hand.length === 0) return false;
      const swapPos = Math.floor(this.rng() * hand.length);

      const out = hand[swapPos];
      hand[swapPos] = card;
      ohand[oi] = out;
      return true;
    }
    return false;
  }

  _repairForceLacks(p, idx, card) {
    const hand = p.hands[idx] || [];
    const pos = hand.indexOf(card);
    if (pos === -1) return true;

    for (let t = 0; t < this.repairAttempts; t++) {
      if (p.deck.length > 0) {
        const di = Math.floor(this.rng() * p.deck.length);
        const inCard = p.deck[di];
        p.deck[di] = card;
        hand[pos] = inCard;
        return true;
      }

      const other = this._randomOtherIndex(p.hands.length, idx);
      if (other === -1) return false;
      const ohand = p.hands[other] || [];
      if (ohand.length === 0) continue;

      const oi = Math.floor(this.rng() * ohand.length);
      const swapIn = ohand[oi];
      if (swapIn === card) continue;

      ohand[oi] = card;
      hand[pos] = swapIn;
      return true;
    }
    return false;
  }

  // ------------ Resampling / rejuvenation ------------

  _effectiveSampleSize() {
    if (!this.particles || this.particles.length === 0) return 0;
    let sum = 0, sumSq = 0;
    for (const p of this.particles) {
      sum += p.w;
      sumSq += p.w * p.w;
    }
    return sumSq > 0 ? (sum * sum) / sumSq : 0;
  }

  _resampleSystematic() {
    const N = this.N;
    const total = this._weightSum();
    if (total <= 0) return;

    // CDF
    const cdf = new Array(N);
    let acc = 0;
    for (let i = 0; i < N; i++) {
      acc += this.particles[i].w / total;
      cdf[i] = acc;
    }

    const newParts = [];
    let u = (this.rng() / N);
    let i = 0;
    for (let j = 0; j < N; j++) {
      while (u > cdf[i] && i < N - 1) i++;
      const src = this.particles[i];
      newParts.push({
        hands: this._deepCopyHands(src.hands),
        deck: src.deck.slice(),
        w: 1.0
      });
      u += 1 / N;
    }
    this.particles = newParts;
  }

  _rejuvenate(rate = 0.1) {
    if (!this.particles) return;
    for (const p of this.particles) {
      if (this.rng() > rate) continue;

      const playerCount = p.hands.length;
      if (this.rng() < 0.5 && p.deck.length > 0) {
        // deck <-> opponent hand swap
        const pi = this._randomOtherIndex(playerCount, this.selfId);
        if (pi === -1) continue;
        const hand = p.hands[pi] || [];
        if (hand.length === 0) continue;

        const hi = Math.floor(this.rng() * hand.length);
        const di = Math.floor(this.rng() * p.deck.length);
        const tmp = hand[hi];
        hand[hi] = p.deck[di];
        p.deck[di] = tmp;
      } else {
        // opponent hand <-> opponent hand swap
        const a = this._randomOtherIndex(playerCount, this.selfId);
        const b = this._randomOtherIndex(playerCount, this.selfId, a);
        if (a === -1 || b === -1) continue;

        const ha = p.hands[a] || [];
        const hb = p.hands[b] || [];
        if (ha.length === 0 || hb.length === 0) continue;

        const ia = Math.floor(this.rng() * ha.length);
        const ib = Math.floor(this.rng() * hb.length);
        const tmp = ha[ia];
        ha[ia] = hb[ib];
        hb[ib] = tmp;
      }
    }
  }

  // ------------ Helpers ------------

  _normalizeIfTiny() {
    const sum = this._weightSum();
    if (sum > 0 && sum < 1e-140) {
      for (const p of this.particles) p.w *= 1e140;
    }
  }

  _normalizeWeights() {
    const sum = this._weightSum();
    if (sum <= 0) return;
    for (const p of this.particles) p.w /= sum;
  }

  _weightSum() {
    let s = 0;
    if (!this.particles) return 0;
    for (const p of this.particles) s += p.w;
    return s;
  }

  _shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      const t = arr[i];
      arr[i] = arr[j];
      arr[j] = t;
    }
  }

  _deepCopyHands(hands) {
    const out = new Array(hands.length);
    for (let i = 0; i < hands.length; i++) out[i] = (hands[i] || []).slice();
    return out;
  }

  _fingerprintRevealed(gameHistory) {
    const rc = gameHistory?.revealedCards;
    if (!Array.isArray(rc)) return "none";
    return rc.map(a => (Array.isArray(a) ? a.join(",") : "")).join("|");
  }

  _coerceToIndex(gameState, maybeId) {
    if (typeof maybeId === "number") return maybeId;
    const players = gameState.players || [];
    for (let i = 0; i < players.length; i++) {
      if (players[i] && players[i].id === maybeId) return i;
    }
    return -1;
  }

  _randomOtherIndex(playerCount, forbidden, alsoForbidden = null) {
    const cand = [];
    for (let i = 0; i < playerCount; i++) {
      if (i === forbidden) continue;
      if (alsoForbidden !== null && i === alsoForbidden) continue;
      cand.push(i);
    }
    if (cand.length === 0) return -1;
    return cand[Math.floor(this.rng() * cand.length)];
  }

  _requiredCardForAction(action) {
    switch (action) {
      case "tax": return "duke";
      case "assassinate": return "assassin";
      case "steal": return "captain";
      case "exchange": return "ambassador";
      default: return null;
    }
  }
}