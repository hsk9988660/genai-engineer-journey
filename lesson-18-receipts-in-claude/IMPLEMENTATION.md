# Lesson 18: implementation guide

This explains **how the code works**: the idea each module adds, what a tool call writes, what verification does, what an attacker can and cannot do, and how to change it. The [README](README.md) says why the lesson exists, what was measured and what is not known. Functions are named instead of numbered by line so this guide does not go stale.

---

## 1. The idea, in the order the code builds it

A receipt is a signed record of one thing an agent did. Each layer below fixes a weakness of the one before.

| Layer | Module | What it adds | The weakness it fixes |
|---|---|---|---|
| 1 | `canon.py` | The same data always becomes the same **bytes** | A signature covers bytes. Two programs that format the same JSON differently would disagree about validity |
| 2 | `keys.py` | Signing keys with **roles**, and a **registry the verifier pins** | A receipt that carries its own key proves only that it agrees with itself |
| 3 | `receipt.py` | What is signed (`kid` and `alg` inside the signed bytes), and **salted commitments** | A plain hash of guessable arguments can be confirmed by guessing |
| 4 | `recorder.py` | One ordered, durable, locked chain; **intent before result** | A receipt written only after the call is missing if the process dies in between |
| 5 | `verify.py` | Verification with a **specific reason** for every failure | "It failed" does not tell an auditor what went wrong |
| 6 | `merkle.py`, `checkpoint.py` | **Signed checkpoints** by a different key | A chain cannot show that its END was cut off |
| 7 | `approval.py` | A human's signature bound to the **exact action**, spent **once** | An action receipt never says a person authorised anything |
| 8 | `gateway.py` | Wiring around a Claude agent's tools | Nothing so far touches a real agent |

---

## 2. A tool call, from start to end

This is the heart of the lesson. Take `executor.execute("refund", fn, {"user_id": "cust-42", "order": "A1", "amount": 4000}, approval)`.

### Where it starts
`GuardedExecutor.execute` in `approval.py`. (For a tool that does not change the world, it goes straight to `ToolRecorder.call`.)

### What happens, in order

1. **Is this an effect tool?** If the tool is in `effect_tools` and there is no approval, it raises `ApprovalRefused("approval_required")`. **Nothing is written**, because nothing was attempted.
2. **Is the approval acceptable?** `verify_approval` checks, in this order: shape; signature against the **pinned approver registry** (key status judged at *now*, so a key rotated out since is refused); the action in the approval hashes to its own `action_digest`; the issue time is not in the future; it has not expired; the policy version still matches; and the approval is for **exactly** this tool and these arguments. Each failure has its own reason.
3. **Spend it.** `ApprovalLedger.reserve(hash)` inserts a row. SQLite's primary key makes this atomic, so of 8 threads exactly one gets `True`. Everyone else gets `already_used`, `in_progress` or `indeterminate_outcome`.
4. **Write the intent.** `ToolRecorder.call` creates a fresh random salt, computes `args_commit = HMAC(salt, canonical(args))`, and `ChainWriter.append` writes an **intent receipt**: under a lock it assigns the next `seq`, sets `prev` to the hash of the previous receipt, stamps `iat`, signs, and appends one line to the file with flush and fsync. The salt goes to the `SaltStore`, not the file.
5. **Run the tool.** `fn(**args)`.
6. **Write the result.**
   - it returned: a **result receipt**, `outcome="success"`, with a salted commitment to the result
   - it raised an ordinary exception: `outcome="failure"` with the exception's type name, then the exception is re-raised
   - it raised `Indeterminate` (the outside world may or may not have acted): `outcome="indeterminate"`, re-raised
   - the **process died** (`BaseException`): **nothing is written**. The intent is on disk with no result
7. **Close the books on the approval.** The ledger row becomes `completed`, `failed` or `indeterminate`. A crash leaves it `reserved`.

### Where it ends
The caller gets the tool's result (or the exception). On disk there are two linked receipts, and the ledger records how the approval was spent.

### What the receipts look like
Values shortened; the field names are the real ones.

```json
{"type": "agent.tool_intent.v2", "alg": "EdDSA", "kid": "k_4f1c...", "iat": 1800000000,
 "receipt_id": "9a3e...", "agent_id": "support-bot", "run_id": "run-1",
 "agent_version": "v-3f9a2c", "model_id": "claude-sonnet-5-5",
 "seq": 0, "prev": null,
 "tool": "refund", "args_commit": "c1:7be0...", "policy_id": "refund-policy", "policy_hash": "sha256:...",
 "approval_ref": "sha256:...", "action_digest": "sha256:...",
 "sig": "pF9..."}

{"type": "agent.tool_result.v2", ..., "seq": 1, "prev": "<hash of the intent>",
 "intent_ref": "<hash of the intent>", "tool": "refund", "outcome": "success",
 "result_commit": "c1:02d4...", "error_type": null, "sig": "..."}
```

Everything except `sig` is signed, including `kid` and `alg`. There is no public key in a receipt.

---

## 3. A verification, from start to end

`ChainVerifier.verify(receipts, ...)` walks the list once.

For each receipt:
1. **Shape.** Required fields and types (a boolean is not an integer). Wrong shape: `malformed`, and the walk continues.
2. **Kind.** Anything other than an intent or result: `unexpected_type`.
3. **Signature**, against the pinned agent registry: `bad_signature`, `unknown_key`, `wrong_role`, `key_expired`, `key_revoked`, `key_not_yet_valid`, `algorithm`. A key that was revoked *after* the receipt's claimed time still verifies but adds the warning `signed_by_key_revoked_later`.
4. **Position.** `seq` must equal the index (`sequence_gap` or `duplicate_sequence`).
5. **Link.** `prev` must equal the hash of the receipt before (`broken_link`).
6. **Identity.** Agent and run id must not change (`mixed_identity`).
7. **Results and intents.** A result's `intent_ref` must name an earlier intent (`orphan_result`), for the same tool (`tool_mismatch`), used once (`duplicate_result`).
8. **Time** (warnings only): going backwards, or in the future.

After the walk:
- **Unresolved intents** are collected: intents with no result.
- **Pinned head / minimum length:** `head_mismatch`, `truncated`.
- **Checkpoints:** each is verified under the **checkpoint registry**. A checkpoint committing to more receipts than are presented is `truncated_vs_checkpoint`; a different receipt at that position is `forked_vs_checkpoint`; a different Merkle root is `merkle_mismatch`. Checkpoints for another run are ignored. Checkpoints must also be linked and in sequence, and a pinned "latest" checkpoint that is not in the history is `checkpoint_rollback_or_fork`.
- **Approvals** (if supplied): every intent that cites an approval must have it supplied (`approval_missing`), still acceptable (`approval_<reason>`), covering the same action digest (`approval_digest_substitution`) and tool, and used by only one intent (`approval_reused`).

The report keeps two questions apart:
- `valid`: **nothing is wrong**
- `complete`: valid **and** every intent has a result. An unresolved intent is not an error. It is "outcome unknown".

---

## 4. Module reference

### `canon.py`
`canonicalize(value) -> bytes` and `digest(value)`. Objects have string keys sorted by **UTF-16 code units** (not code points: U+1F600 sorts before U+E000). Strings are escaped one way. It **refuses** floats, integers beyond +/-(2^53-1), lone surrogates, non-string keys, unknown types, nesting beyond 32 levels and more than 1 MB. It is a *subset* of RFC 8785, not a conformant implementation.

### `keys.py`
`Signer` (wraps a private key and a role; production would call a key vault), `KeyRegistry` (kid -> public key, role, validity window, revocation), `KeyProblem(reason)`. `resolve` returns the key or says exactly why not. **Revocation and validity are judged against the time the signer wrote**, which someone with a stolen old key can backdate.

### `receipt.py`
`sign`, `verify_signature`, `receipt_hash` (the hash covers the signature), `shape_problems`, `commit` / `open_commit` (HMAC-SHA256 with a salt as the key), `to_tree` (turns floats, Decimals and bytes into tagged strings instead of losing information), strict base64url.

### `recorder.py`
`ChainWriter` (locked, durable, resumable), `read_chain` (unreadable lines stay visible), `SaltStore`, `ToolRecorder.call` (intent, run, result), `Indeterminate`.

### `verify.py`
`ChainVerifier`, `VerifyReport`, `Finding`. Three registries on purpose: agent keys, checkpoint keys, approver keys.

### `merkle.py` and `checkpoint.py`
An RFC 6962-style tree (leaves hashed with prefix 0x00, inner nodes with 0x01), inclusion proofs, and signed checkpoints that commit to length, head and Merkle root, linked to each other.
**A property worth knowing:** an inclusion proof does not prove the tree's *size*. A leaf in the left half cannot tell whether its right sibling covers one leaf or two. The size must come from the signed checkpoint, which signs `length` with the root.

### `approval.py`
`issue_approval`, `verify_approval`, `ApprovalLedger` (SQLite; reserve, complete, `mark_unfinished_indeterminate`), `GuardedExecutor`, `check_intents_against_approvals`.

### `gateway.py`
`wrap_impls` turns lesson 16's `impl(user_id, **args)` tools into receipted, approval-guarded ones. **The user id is part of the recorded action**, so an approval for one customer cannot cover another. `ApprovalBroker` holds approvals placed by the system (a person clicking "approve"), never by the model. `run_turn` is a minimal Claude tool loop to show the wiring.

---

## 5. What an attacker can and cannot do

| Attack | What stops it | What remains |
|---|---|---|
| Edit a field in a receipt | The signature fails (`bad_signature`) | Nothing |
| Re-sign an edited receipt with their own key | The key is not in the pinned registry (`unknown_key`) | If they can get **your** registry changed, nothing helps |
| Copy the real key id and make up a signature | `bad_signature` | Nothing |
| Delete or reorder a receipt in the middle | `sequence_gap`, `broken_link` | Nothing |
| Cut off the **end** of the chain | A pinned head or a signed checkpoint | **Without one of those, undetectable** |
| Rewrite history with the agent's real key | A checkpoint signed by a *different* key (`forked_vs_checkpoint`) | If the checkpoint is not stored somewhere they cannot reach, they rewrite that too |
| Reuse one approval | The ledger (`already_used`) | The ledger is a local database: it must be shared or durable |
| Use an approval for a bigger amount | `action_mismatch` | Nothing |
| Spend one approval from two threads | Atomic reservation | Nothing |
| Guess arguments from a receipt | Salted commitment | Anyone who gets the **salt** can check content |
| A **compromised signing process** | Nothing | It signs lies. A receipt shows what its issuer claims happened |

---

## 6. Concurrency, durability and crashes

- `ChainWriter.append` holds one lock around assigning `seq` and `prev`, signing, writing and updating the head, so two threads can never share a sequence number.
- A receipt that `append` has returned has been written, flushed and (by default) `fsync`ed. **I cannot test fsync durability**: no test pulls the power.
- A crash between intent and result leaves an unresolved intent. A crash inside the tool leaves the approval `reserved`; at startup call `mark_unfinished_indeterminate()` to turn those into "outcome unknown". The code **never retries** an indeterminate action automatically.
- A resumed `ChainWriter` reads (does not verify) the tail of the file, so it continues the sequence. It refuses to append to a file whose last line is not a receipt.
- The ledger is SQLite with one process-wide lock: atomic across threads in one process. Several processes sharing a file were not tested.

---

## 7. Key handling (what the code does and does not do)

- Keys have **roles**: an agent key cannot sign an approval or a checkpoint, and `sign` refuses.
- A registry entry has a validity window and can be revoked. Rotating a key means registering the new one and setting `valid_to` or `revoke` on the old.
- **Nothing here stores a private key safely.** `Signer.generate` keeps it in memory. Use a key vault or HSM for real.
- **Nothing here publishes a registry.** How a verifier gets the right public keys, and trusts them, is the part this code cannot do for you.

---

## 8. How to extend it

- **Record a new kind of event.** Add a type and shape in `receipt.SHAPES` and `ROLE_OF`, a verifier rule, and tests for the attack you are guarding against.
- **Record more per call.** Add fields to the intent or result shape. Anything not in the signed payload is not protected.
- **Anchor checkpoints.** Write `make_checkpoint(...)` output to something the agent's host cannot rewrite (a transparency log, another system, a timestamp service) and keep its hash for `pinned_latest_checkpoint`. This is where truncation protection actually comes from.
- **Selective disclosure.** `merkle.inclusion_proof` gives a short proof that one receipt is under a signed root, without showing the others.
- **Different algorithm.** Accept a new `alg` only with a full profile (key encoding, canonical bytes, downgrade policy). Changing a label is not a migration.

---

## 9. Design decisions and what was rejected

- **No key in the receipt.** The most important change from the lesson's layout. A key that arrives with the data it vouches for is not a trust anchor.
- **Intent first, result second.** Costs two receipts per call. Buys a visible record of "started, outcome unknown".
- **One chain per agent run, not per tool.** Per-tool chains cannot show that a whole tool's record was deleted.
- **Keyed commitments, not plain hashes.** Costs a salt to manage. Buys protection against guessing.
- **A restricted canonical form that refuses floats.** Costs convenience (amounts as integer cents or strings). Buys not having to get number formatting exactly right.
- **`valid` and `complete` are separate.** An honest crash must not look like tampering, and tampering must not look like a crash.
- **Rejected: trusting the receipt's timestamp.** It is a claim by the signer.
- **Rejected: a full RFC 8785 implementation here.** Its float rules are exactly where a hand-written version is most likely to be subtly wrong.

---

## 10. Running and testing

`python -m pytest -q` in this folder (a few seconds; no model, no key, no network). `python harness.py` prints the seven scenarios. The only dependency beyond the rest of the repo is `cryptography` (Ed25519).

Tests use deterministic fakes: a seeded signer, a fixed clock, scripted Claude replies. A simulated crash is an exception derived from `BaseException`, so nothing in the recorder's `except Exception` can swallow it. **No test kills a real process, and none uses a real key vault, a real transparency log or the real API.**
