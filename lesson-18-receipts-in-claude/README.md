# Lesson 18: signed receipts for agent actions

Lesson 18 of [ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) is titled "Securing AI Agents" but is really about one thing: **cryptographic receipts**, signed records of what an agent did, so an auditor can check a log without trusting whoever kept it. It is the most careful lesson in the course about its own limits. This folder rebuilds it on the same primitives (Ed25519 signatures over canonical JSON, hash-chained receipts, pinned keys, signed human approvals) and closes the gaps the lesson leaves open.

> **A receipt proves that a particular key signed particular content. It does not prove the content was true, that the action happened, or that the record is complete.** Everything useful about receipts comes from being exact about that boundary, and from the controls around it: a verifier that pins the right keys, a chain that shows what is missing, and approvals that can be spent once.

For how the code works (a tool call from start to end, a verification from start to end, what an attacker can and cannot do), read [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Run

```bash
pip install anthropic pydantic pytest cryptography
python -m pytest -q          # 222 offline tests, no model, no key, no network, a few seconds
python harness.py            # the seven scenarios below
```

## What is built

| File | What it is |
|---|---|
| `recorder.py` | **The core.** A locked, durable, resumable chain writer, and a recorder that writes an **intent** receipt before a tool runs and a **result** receipt after. A crash leaves a visible unresolved intent. |
| `verify.py` | A chain verifier with a **specific reason** for each failure, and two separate questions: is it `valid`, and is it `complete`. |
| `receipt.py` | What is signed (`kid` and `alg` inside the signed bytes, **no public key in the receipt**), model and agent version on every receipt, and **salted commitments** instead of plain hashes. |
| `keys.py` | Signing keys with **roles** (agent, approver, checkpoint) and a registry the verifier pins, with validity windows and revocation. |
| `canon.py` | Canonical JSON: the same data always becomes the same bytes. A restricted subset of RFC 8785 that refuses floats. |
| `merkle.py`, `checkpoint.py` | A Merkle tree and **signed checkpoints** by a separate key, so a cut-off end or a rewritten history can be detected. |
| `approval.py` | A human's signature bound to the **exact** action, a one-time ledger, and an executor that will not run a world-changing tool without a valid, unspent approval. |
| `gateway.py` | Receipts and approvals around the same `impl(user_id, **args)` tools lesson 16 uses, with a minimal Claude tool loop. |
| `harness.py` | The lesson's patterns versus these, on seven scenarios. |

## The rules the code enforces
1. **A receipt carries no key.** It names its signing key by id, inside the signed bytes, and the verifier looks that id up in a registry it pinned. A receipt that arrives with its own key proves only that it agrees with itself.
2. **Keys have roles.** An agent key cannot sign an approval or a checkpoint, and each role has its own registry.
3. **Canonical bytes.** Receipts are signed over a canonical form, so reformatting cannot invalidate a signature or let a different document pass. Floats are refused: use integers or strings.
4. **Every receipt says what decided.** Agent id, run id, **agent version**, **model id** and a policy **hash**, not just a policy name.
5. **Arguments and results are salted commitments**, not plain hashes. Anyone holding the receipt cannot confirm a guess. The salt is kept outside the chain and revealed only to someone who should be able to check.
6. **Intent first, result second.** A crash between them leaves an intent with no result, which the verifier reports as *unresolved*, different from nothing at all.
7. **One chain per agent run.** Sequence numbers and previous-receipt hashes are assigned under a lock, so threads cannot collide, and deleting a receipt from the middle is a visible gap.
8. **Every failure has its own name:** `bad_signature`, `unknown_key`, `wrong_role`, `key_expired`, `key_revoked`, `sequence_gap`, `broken_link`, `orphan_result`, `forked_vs_checkpoint` and so on, so an auditor can tell what happened.
9. **`valid` and `complete` are different.** An honest crash must not look like tampering, and tampering must not look like a crash.
10. **Checkpoints are signed by a different key**, link to each other, commit to a Merkle root, and a verifier can be given the hash of the newest one it knows, so an older history is refused.
11. **An approval binds to the exact action.** The approver signs the full canonical action and its digest. A real approval for a different action, an expired one, one under an old policy, one from a rotated-out key, and an altered one are each refused with their own reason.
12. **An approval is spent once.** It is reserved atomically before the tool runs, so two threads cannot both use it. A crash leaves it reserved, and a restart turns that into "outcome unknown", which is never retried automatically.
13. **A failed action still spends the approval.** A new one is needed.
14. **The user is part of the recorded action,** so an approval for one customer cannot cover another.

## Measured results

`python harness.py`. "The lesson's pattern" is my restatement of the notebook's behaviour on the same primitives, not its exact code, and I did not run the notebook.

| Scenario | The lesson's pattern | Here |
|---|---|---|
| A receipt re-signed with the attacker's own key and that key embedded | verifying against the **embedded key accepts it**. Against a pinned key it is refused | a whole chain signed by an unregistered key: refused (`unknown_key`). One that copies the real key id: refused (`bad_signature`) |
| Who can forge a log? | an HMAC chain: a verifier who holds the key **can forge a chain that verifies** | Ed25519, verifier holds only public keys: its forgery does **not** verify |
| The last 4 of 12 receipts are removed | the shortened chain **still verifies** | alone: it still verifies and nothing says otherwise. With a pinned head: caught. With a signed checkpoint: caught |
| The process dies after the booking, before the receipt | **1 effect, 0 receipts**: no trace at all | 1 effect, 1 receipt (the intent), `valid` but **not complete**, intent 0 reported as unresolved |
| Recovering arguments from a hash (a search space of 11,400) | the plain hash is matched after **4,371 guesses** | the salted commitment is never matched. The owner can open it, a wrong salt cannot |
| An "approved" flag on the call | 3 refunds execute (4,000, 400,000, 4,000) | **1** executes. The others: `approval_required`, `already_used`, `action_mismatch`, `unknown_key`, `stale_policy`, `expired`, `key_revoked` |
| 8 threads spend one approval at once | | **1 of 8** executes |
| A whole tool's receipts are deleted | a chain per tool: the rest still verify, deletion **undetected**, and no receipt names the model | one chain: `sequence_gap` and `broken_link`; every receipt names the model and agent version |

**How to read this honestly.**
- Three rows are about **Ed25519 and HMAC themselves** behaving as designed. They test that the code uses them correctly, not that the cryptography is sound.
- The truncation row is the important one: **a chain cannot show that its end is missing.** The protection comes from a pinned head or a checkpoint stored where the agent's host cannot rewrite it, and this code does not store or anchor a checkpoint anywhere.
- The crash is *simulated* (an exception that nothing in the recorder can catch). No process was killed and no power was cut.

## Findings from building it
1. **The verifier crashed on a receipt of the wrong kind.** An approval receipt is a valid shape but has no sequence number, so `verify` raised `KeyError` instead of reporting `unexpected_type`. A test found it, and it now reports it.
2. **My first "deleted a tool's receipts" scenario deleted the last tool's receipts, and reported nothing.** That is just truncation, which the chain cannot show. I moved the deletion to the middle, and kept the truncation limit as its own row.
3. **Merkle inclusion proofs do not prove the tree's size.** A leaf in the left half cannot tell whether its right sibling covers one leaf or two, so a size-9 proof still verifies as "size 10". The size has to come from the signed checkpoint, which signs the length with the root. A test documents this.
4. **Mutation checking found one real gap:** a negative or too-large proof index walked the same path as a valid one and would verify a proof built for another position. My test had used an empty proof, which hid it.
5. **Three guards survive deletion and I believe they are redundant, not gaps.** The role check in `issue_approval` and `make_checkpoint` is backed by `sign()` refusing the wrong role, and the key-length check by the cryptography library rejecting a wrong-length key.
6. **A test assumed a file exists when nothing had been written.** When an approval is refused, no chain file is created at all, which is the point.

## How I checked the tests can fail
I mutation-checked 115 behaviors in a scratch copy (each run in parallel, every mutant checked to compile): canonical encoding rules, Merkle construction, every key and role check, signature and shape checks, commitments, the chain writer, every recording rule, every approval refusal, the ledger, executor, checkpoint rules, each verifier check, and the gateway. **111 were caught on the first run. Of the 4 survivors, one was a real test gap (now caught, 112 of 115) and 3 are the redundant guards above.** That means these changes are detected, not that the suite is complete. I did not try deleting `fsync`, because no test can observe durability.

## What is NOT known
- **I did not run the lesson's notebook.** My critique is from reading it. In particular I believe its `verify_receipt` uses `BadSignatureError` without importing it, which would raise `NameError` on a bad signature. I confirmed only the **Python behaviour** with a stand-in snippet (an undefined name in an `except` clause raises `NameError`), not the notebook itself. Its saved output shows the tamper case returning `False`, so run the notebook top to bottom to settle it.
- **This is not a conformant RFC 8785 implementation** (no floats), and I did not cross-check it, or the Merkle tree, against published test vectors or another library.
- **The cryptography is the `cryptography` package's.** I did not audit it, and I tested only that this code uses it correctly.
- **No real key vault or HSM, transparency log or timestamp authority.** Keys live in memory here, and anchoring checkpoints is the part that actually gives truncation protection.
- **Durability is untested.** `fsync` is called, but no test pulls the power or kills a real process.
- **Several processes sharing the chain file or the ledger were not tested.**
- **Performance is unmeasured.** Each receipt costs a signature and an `fsync`, and every call writes two.
- **No real Claude API call:** `run_turn` is tested with scripted replies.
- **Windows only**, with one version of the `cryptography` package.

## Limits of the design
- **A compromised signing process signs lies.** A receipt shows what its issuer claims happened. Executor-side evidence is a separate control.
- **Truncation of the end is undetectable** without a pinned head or an anchored checkpoint.
- **Key validity and revocation use the time the signer wrote.** Someone with a stolen old key can backdate. The `signed_by_key_revoked_later` warning and checkpoints narrow this, they do not remove it.
- **The salt store is in memory.** Lose the salts and the commitments can never be opened. Persisting and protecting them is up to you.
- **Salts are secrets:** whoever holds one can check that content. Deleting a salt makes its commitment unopenable, which may or may not satisfy a deletion requirement. That is a legal question I have not answered.
- **One chain per run** needs a decision about what a run is, and gives no ordering across runs.
- **One global write lock** limits throughput.
- **An approval proves a key signed, not that a person was present.** Binding the approver key to a human and to WebAuthn-style presence is out of scope.
- **The ledger is local SQLite.** Several servers need a shared, atomic one.
- **Floats are refused** in the canonical form. Tool arguments and results are converted before being committed to, but anything in the receipt itself must be an integer or a string.

## Exercises
1. Run the lesson's notebook top to bottom in a fresh kernel. Does `verify_receipt` raise on a bad signature?
2. Store checkpoints somewhere the agent's host cannot write (a second machine, an append-only bucket). Then try to truncate the chain.
3. Replace `Signer` with a call to a real key vault. What changes in `ChainWriter`?
4. Add a signed "model call" receipt that commits to the prompt and the response. What do you gain, and what privacy do you give up?
5. Kill a real process between an intent and a result. What does the verifier say, and what should an operator do with an `indeterminate` approval?
