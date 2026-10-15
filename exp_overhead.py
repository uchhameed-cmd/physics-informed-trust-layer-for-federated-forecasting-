"""Measured cost of the security layers: post-quantum channel per model update, replay
protection, and a PBFT-style permissioned ledger simulated in-process.
Results -> results/overhead.json"""
import hashlib, json, os, struct, time
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives import hashes, serialization
from pqcrypto.kem import ml_kem_768
from pqcrypto.sign import ml_dsa_44

from qtrust.model import Forecaster, n_params

RES = Path(__file__).parent / "results"
TRIALS = 300


def timeit(fn, n=TRIALS):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3), float(np.quantile(ts, 0.95) * 1e3)


def header(cid, rnd, nonce, digest):
    return struct.pack(">HI", cid, rnd) + nonce + digest


class Receiver:
    """Verifies identity, signature, freshness (round id + nonce) and payload integrity."""
    def __init__(self, pks):
        self.pks, self.seen, self.round = pks, set(), 0

    def accept(self, msg):
        cid, rnd = struct.unpack(">HI", msg["hdr"][:6])
        nonce = msg["hdr"][6:22]
        if rnd != self.round or (cid, nonce) in self.seen:
            return False, "stale-or-replayed"
        try:
            ml_dsa_44.verify(self.pks[cid], msg["hdr"], msg["sig"])
        except Exception:
            return False, "bad-signature"
        if hashlib.sha3_256(msg["ct"]).digest() != msg["hdr"][22:]:
            return False, "payload-mismatch"
        self.seen.add((cid, nonce))
        return True, "ok"


def main():
    d = n_params(Forecaster())
    payload = os.urandom(4 * d)                                # float32 update
    out = dict(n_params=d, update_bytes=len(payload))
    # ---- post-quantum channel
    kpk, ksk = ml_kem_768.generate_keypair()
    spk, ssk = ml_dsa_44.generate_keypair()
    ct_kem, ss = ml_kem_768.encrypt(kpk)
    aes = AESGCM(ss[:32]); nonce12 = os.urandom(12)
    ct = aes.encrypt(nonce12, payload, None)
    hdr = header(3, 7, os.urandom(16), hashlib.sha3_256(ct).digest())
    sig = ml_dsa_44.sign(ssk, hdr)
    out["pqc"] = {
        "ml_kem_768_keygen_ms": timeit(ml_kem_768.generate_keypair),
        "ml_kem_768_encaps_ms": timeit(lambda: ml_kem_768.encrypt(kpk)),
        "ml_kem_768_decaps_ms": timeit(lambda: ml_kem_768.decrypt(ksk, ct_kem)),
        "ml_dsa_44_sign_ms": timeit(lambda: ml_dsa_44.sign(ssk, hdr)),
        "ml_dsa_44_verify_ms": timeit(lambda: ml_dsa_44.verify(spk, hdr, sig)),
        "aes256gcm_encrypt_update_ms": timeit(lambda: aes.encrypt(os.urandom(12), payload, None)),
        "aes256gcm_decrypt_update_ms": timeit(lambda: aes.decrypt(nonce12, ct, None)),
        "sha3_update_ms": timeit(lambda: hashlib.sha3_256(ct).digest()),
        "sizes_bytes": dict(kem_pk=len(kpk), kem_ct=len(ct_kem), dsa_pk=len(spk), dsa_sig=len(sig),
                            header=len(hdr), aead_overhead=len(ct) - len(payload)),
    }
    # ---- classical baseline (X25519 + ECDSA P-256)
    xa = X25519PrivateKey.generate(); xb = X25519PrivateKey.generate()
    eck = ec.generate_private_key(ec.SECP256R1()); ecs = eck.sign(hdr, ec.ECDSA(hashes.SHA256()))
    out["classical"] = {
        "x25519_exchange_ms": timeit(lambda: xa.exchange(xb.public_key())),
        "ecdsa_p256_sign_ms": timeit(lambda: eck.sign(hdr, ec.ECDSA(hashes.SHA256()))),
        "ecdsa_p256_verify_ms": timeit(lambda: eck.public_key().verify(ecs, hdr, ec.ECDSA(hashes.SHA256()))),
        "sizes_bytes": dict(x25519_pk=32, ecdsa_sig=len(ecs),
                            ecdsa_pk=len(eck.public_key().public_bytes(serialization.Encoding.X962,
                                                                       serialization.PublicFormat.CompressedPoint))),
    }
    # per-round, per-client added bytes (session key re-used for a round; header + signature)
    pq = out["pqc"]["sizes_bytes"]
    out["per_update_overhead_bytes_pqc"] = pq["header"] + pq["dsa_sig"] + pq["aead_overhead"]
    out["per_update_overhead_bytes_classical"] = pq["header"] + out["classical"]["sizes_bytes"]["ecdsa_sig"] + pq["aead_overhead"]
    out["per_session_setup_bytes_pqc"] = pq["kem_pk"] + pq["kem_ct"]
    # ---- functional freshness test: 17 clients, 40 rounds, every message replayed once
    N, R = 17, 40
    keys = [ml_dsa_44.generate_keypair() for _ in range(N)]
    rx = Receiver({i: keys[i][0] for i in range(N)})
    acc = rej_replay = rej_tamper = 0
    prev = []
    for r in range(R):
        rx.round = r
        cur = []
        for i in range(N):
            c = os.urandom(256)
            hd = header(i, r, os.urandom(16), hashlib.sha3_256(c).digest())
            m = dict(hdr=hd, sig=ml_dsa_44.sign(keys[i][1], hd), ct=c)
            ok, _ = rx.accept(m); acc += ok; cur.append(m)
        for m in cur + prev:                                   # same-round and previous-round replays
            ok, why = rx.accept(m); rej_replay += (not ok)
        t = dict(cur[0]); t["ct"] = os.urandom(256)            # tampered payload
        rx.seen.discard(tuple()); ok, why = rx.accept(dict(t, hdr=header(0, r, os.urandom(16), t["hdr"][22:]), sig=t["sig"]))
        rej_tamper += (not ok)
        prev = cur
    out["freshness_test"] = dict(accepted=acc, expected=N * R, replays_rejected=rej_replay,
                                 replays_sent=N * R + N * (R - 1), tampered_rejected=rej_tamper, tampered_sent=R)
    # ---- PBFT-style ledger, n=4 validators (f=1): CPU per block measured, network delay modelled
    n_val, f_tol = 4, 1
    vkeys = [ml_dsa_44.generate_keypair() for _ in range(n_val)]
    blocks = []
    prev_hash = b"\x00" * 32
    msgs = [dict(hdr=header(i, 0, os.urandom(16), os.urandom(32))) for i in range(N)]
    for i in range(N):
        msgs[i]["sig"] = ml_dsa_44.sign(keys[i][1], msgs[i]["hdr"])
    for b in range(50):
        t0 = time.perf_counter()
        # each validator re-verifies the 17 client signatures before voting
        for v in range(n_val):
            for i in range(N):
                ml_dsa_44.verify(keys[i][0], msgs[i]["hdr"], msgs[i]["sig"])
        body = json.dumps(dict(round=b, prev=prev_hash.hex(),
                               commits=[hashlib.sha3_256(m["hdr"]).hexdigest() for m in msgs],
                               trust=[round(float(x), 3) for x in np.random.rand(N)],
                               model=hashlib.sha3_256(os.urandom(64)).hexdigest())).encode()
        bh = hashlib.sha3_256(body).digest()
        votes = []
        for phase in ("prepare", "commit"):                    # 2 all-to-all voting phases
            for v in range(n_val):
                s_ = ml_dsa_44.sign(vkeys[v][1], phase.encode() + bh)
                votes.append(s_)
                for u in range(n_val):
                    if u != v:
                        ml_dsa_44.verify(vkeys[v][0], phase.encode() + bh, s_)
        cpu = time.perf_counter() - t0
        prev_hash = bh
        blocks.append(dict(cpu_ms=cpu * 1e3, body_bytes=len(body), vote_bytes=sum(len(x) for x in votes)))
    cpu = np.array([b_["cpu_ms"] for b_ in blocks])
    out["ledger"] = dict(validators=n_val, f=f_tol, quorum=2 * f_tol + 1, messages_per_block=1 + 2 * n_val * (n_val - 1) + n_val,
                         cpu_ms_median=float(np.median(cpu)), cpu_ms_p95=float(np.quantile(cpu, 0.95)),
                         block_body_bytes=blocks[0]["body_bytes"], vote_bytes=blocks[0]["vote_bytes"],
                         modelled_latency_ms={str(rtt): float(np.median(cpu) + 3 * rtt) for rtt in (1, 20, 80)},
                         note="3 message delays (pre-prepare, prepare, commit); RTT values are assumptions")
    (RES / "overhead.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
