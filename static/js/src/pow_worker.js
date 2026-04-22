/*
 * PoW Web Worker — runs in isolation so the SHA-256 grind doesn't freeze
 * the main thread. Spawned from pow_challenge.js via a Blob URL so it
 * inherits no DOM or fetch surface.
 *
 * Kept in plain ES5-compatible syntax (no arrow-fn-in-class, no optional
 * chaining, …) so terser's default ECMA level produces the smallest
 * output and we stay compatible with any browser that can run crypto.
 *
 * C3 #9: extracted from the pow_challenge.py f-string template so the
 * build pipeline can minify it once and Python just embeds the result.
 */

(function () {
  "use strict";

  const K = new Uint32Array([
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
  ]);

  function sha256(msg) {
    const msgLen = msg.length;
    const bitLen = msgLen * 8;
    const padLen = ((msgLen + 9 + 63) & ~63);
    const buf = new Uint8Array(padLen);
    for (let i = 0; i < msgLen; i++) buf[i] = msg.charCodeAt(i);
    buf[msgLen] = 0x80;
    const view = new DataView(buf.buffer);
    view.setUint32(padLen - 4, bitLen, false);

    let h0=0x6a09e667, h1=0xbb67ae85, h2=0x3c6ef372, h3=0xa54ff53a;
    let h4=0x510e527f, h5=0x9b05688c, h6=0x1f83d9ab, h7=0x5be0cd19;
    const w = new Uint32Array(64);

    for (let off = 0; off < padLen; off += 64) {
      for (let i = 0; i < 16; i++) w[i] = view.getUint32(off + i*4, false);
      for (let i = 16; i < 64; i++) {
        const s0 = (w[i-15]>>>7 | w[i-15]<<25) ^ (w[i-15]>>>18 | w[i-15]<<14) ^ (w[i-15]>>>3);
        const s1 = (w[i-2]>>>17 | w[i-2]<<15) ^ (w[i-2]>>>19 | w[i-2]<<13) ^ (w[i-2]>>>10);
        w[i] = (w[i-16] + s0 + w[i-7] + s1) | 0;
      }
      let a=h0,b=h1,c=h2,d=h3,e=h4,f=h5,g=h6,h=h7;
      for (let i = 0; i < 64; i++) {
        const S1 = (e>>>6|e<<26)^(e>>>11|e<<21)^(e>>>25|e<<7);
        const ch = (e&f)^(~e&g);
        const t1 = (h+S1+ch+K[i]+w[i])|0;
        const S0 = (a>>>2|a<<30)^(a>>>13|a<<19)^(a>>>22|a<<10);
        const maj = (a&b)^(a&c)^(b&c);
        const t2 = (S0+maj)|0;
        h=g; g=f; f=e; e=(d+t1)|0; d=c; c=b; b=a; a=(t1+t2)|0;
      }
      h0=(h0+a)|0; h1=(h1+b)|0; h2=(h2+c)|0; h3=(h3+d)|0;
      h4=(h4+e)|0; h5=(h5+f)|0; h6=(h6+g)|0; h7=(h7+h)|0;
    }
    return [h0,h1,h2,h3,h4,h5,h6,h7];
  }

  function sha256hex(msg) {
    const h = sha256(msg);
    let out = "";
    for (let i = 0; i < 8; i++) {
      out += ("00000000" + (h[i]>>>0).toString(16)).slice(-8);
    }
    return out;
  }

  function checkLeadingZeros(hash, bits) {
    const fullBytes = bits >> 3;
    const remBits = bits & 7;
    for (let i = 0; i < fullBytes; i++) {
      const byte = (hash[i>>2] >>> (24 - (i&3)*8)) & 0xFF;
      if (byte !== 0) return false;
    }
    if (remBits > 0) {
      const byte = (hash[fullBytes>>2] >>> (24 - (fullBytes&3)*8)) & 0xFF;
      if ((byte & (0xFF << (8 - remBits))) !== 0) return false;
    }
    return true;
  }

  self.onmessage = function (e) {
    const batches = e.data.batches;
    const totalBatches = batches.length;
    const startTime = Date.now();
    const nonces = [];
    let runningSalt = "";

    function solveBatch(batchIdx) {
      if (batchIdx >= totalBatches) {
        self.postMessage({
          done: true,
          nonces: nonces,
          elapsed: Date.now() - startTime,
        });
        return;
      }

      const batch = batches[batchIdx];
      const prefix = batch.prefix;
      const difficulty = batch.difficulty;
      let nonce = 0;
      const batchSize = 5000;

      function solveChunk() {
        for (let i = 0; i < batchSize; i++) {
          const candidate = prefix + runningSalt + nonce.toString(16);
          const hash = sha256(candidate);
          if (checkLeadingZeros(hash, difficulty)) {
            const foundNonce = nonce.toString(16);
            nonces.push(foundNonce);
            runningSalt = sha256hex(candidate);
            self.postMessage({
              done: false,
              batchDone: true,
              batchIndex: batchIdx,
              totalBatches: totalBatches,
              nonce: foundNonce,
              elapsed: Date.now() - startTime,
            });
            setTimeout(function () { solveBatch(batchIdx + 1); }, 0);
            return;
          }
          nonce++;
        }
        self.postMessage({
          done: false,
          batchDone: false,
          batchIndex: batchIdx,
          totalBatches: totalBatches,
          hashes: nonce,
          elapsed: Date.now() - startTime,
        });
        setTimeout(solveChunk, 0);
      }
      solveChunk();
    }
    solveBatch(0);
  };
})();
