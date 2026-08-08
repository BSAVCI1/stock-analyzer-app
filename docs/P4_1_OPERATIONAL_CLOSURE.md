# P4.1 Operational Closure — EUR Portfolio and Fixed-Notional Sizing

**Project:** Smart Investment Bot
**Phase:** P4.1 — EUR 2,000 account and fixed-notional sizing
**Closure date:** 8 August 2026
**Product mode:** PAPER_ONLY
**Decision:** COMPLETE

## 1. Executive conclusion

P4.1 is complete. The project now has a dedicated EUR 2,000 operational paper account, deterministic fixed-notional sizing, explicit portfolio-risk limits, multi-currency lifecycle economics and a runtime FX provider. The scheduled runtime loads the operational account and its persisted controls successfully. Account cash and ledger reconcile with zero difference.

No broker connection or live-execution capability was introduced.

## 2. Operational portfolio identity

- Operational account: `ACC-749ca5703d214ef0b91f87b825e88849`
- Name: `P4 EUR Paper Portfolio`
- Currency: EUR
- Starting balance: EUR 2,000
- Cash at closure baseline: EUR 2,000
- Reserved cash: EUR 0
- Available cash: EUR 2,000
- Status: ACTIVE
- Open positions: 0
- Pending orders: 0
- Schema version: 8
- Bootstrap database SHA256: `45364e65e2f95bd4b016c8332ed37410fe8ed494d8ce9917546e21792401a22e`

The database hash is a closure-baseline fingerprint. It is expected to change after legitimate future operational events.

## 3. Approved sizing policy

| Control | P4.1 value | Status |
|---|---:|---|
| Sizing mode | `FIXED_NOTIONAL_WITH_RISK_CAP` | PASS |
| Portfolio currency | EUR | PASS |
| Target order value | EUR 100 | PASS |
| Hard order ceiling | EUR 100 | PASS |
| Maximum planned loss | EUR 10 | PASS |
| Maximum open positions | 5 | PASS |
| Maximum invested exposure | EUR 500 | PASS |

The sizing core reduces quantity when notional, risk, fees, available cash, exposure, position-count or quantity-step constraints require it. Whole-share sizing floors to the permitted quantity and does not round above the configured ceiling.

## 4. Multi-currency and runtime evidence

P4.1 introduced explicit quote-currency and FX provenance across the paper lifecycle. Security prices remain in quote currency while account cash, reservations, ledger values and lifecycle P&L are represented in portfolio currency.

Runtime FX resolution is provided by `YahooFXRateProvider`. Same-currency valuation uses exact identity FX. Cross-currency valuation requires a valid rate; the runtime does not silently assume 1:1. Yahoo direct pairs are preferred and inverse pairs may be reciprocated when the direct pair is unavailable.

The execution engine now sizes candidate orders and values portfolio positions in the account portfolio currency.

## 5. Reconciliation and lifecycle acceptance

The operational account passed the closure check with:

- Stored cash: EUR 2,000
- Ledger cash: EUR 2,000
- Difference: EUR 0
- Reconciled: `True`
- Open positions: 0
- Pending orders: 0
- Trading and scheduled-job activity at bootstrap: 0

The acceptance proof was read-only: the operational database SHA256 was unchanged before and after runtime validation.

Automated lifecycle tests additionally cover reservation, fill, close, FX conversion and reconciliation behavior so P4.1 does not require a fabricated operational trade merely to satisfy the phase gate.

## 6. Implementation checkpoints

The P4.1 implementation was delivered incrementally through these checkpoints:

- `1db9d1d` — fixed-notional sizing and FX policy core
- `52cc5c4` — account sizing controls and schema v6
- `ed74d49` — portfolio-currency economics
- `f1659cc` — signal quote-currency persistence
- `c731bc2` — scanner quote-currency propagation
- `00f1b03` — lifecycle FX provenance schema
- `e264b06` — lifecycle FX economics
- `b816ec4` — fixed-notional sizing in the execution engine
- `a61ac29` — Yahoo FX provider wired into runtime

Regression reached 452 passing tests locally and in a clean CI-like environment. The pushed `a61ac29` checkpoint showed a successful GitHub Automated tests #58 run.

## 7. Historical P3 preservation

The original P3 runtime SQLite database was not recovered after the development environment was replaced. P4.1 does not claim otherwise.

The verified P3 release evidence remains preserved in:

- tag `v0.3.0-p3`
- `docs/P3_OPERATIONAL_CLOSURE.md`
- `IMPLEMENTATION_ROADMAP.md`
- versioned product policy historical-account record

Historical P3 account:

`ACC-495a2ae778834fc4a2c14d24e66ef41e`

The new P4 operational database intentionally contains no fabricated or reconstructed P3 scan, execution, job, notification or trading rows.

## 8. P4.1 closure gate

- [x] New EUR 2,000 operational paper account exists.
- [x] Operational account ID is captured.
- [x] Fixed-notional sizing mode is persisted.
- [x] EUR 100 target is enforced.
- [x] EUR 100 hard ceiling is enforced.
- [x] Planned loss is capped at EUR 10.
- [x] Open positions are capped at five.
- [x] Invested exposure is capped at EUR 500.
- [x] Whole-share rounding cannot exceed the ceiling.
- [x] Multi-currency valuation and lifecycle economics are supported.
- [x] Cross-currency FX provenance is persisted.
- [x] Runtime FX resolution is wired.
- [x] Cash and ledger reconcile exactly.
- [x] Regression remains green.
- [x] No live execution or broker connectivity was introduced.

**Final decision: P4.1 COMPLETE.**

## 9. Controlled transition to P4.2

P4.2 adds the IBKR reference cost profile without connecting an IBKR account.

The next phase must model:

1. Versioned IBKR commission assumptions.
2. Minimum commission and applicable exchange/regulatory costs.
3. Fractional-share and minimum-notional behavior.
4. FX conversion costs.
5. Round-trip total cost.
6. Net reward-to-risk after costs.
7. Rejection of EUR 100 trades that are uneconomic after realistic costs.

IBKR remains a reference cost model and optional manual execution venue only. No credentials or broker connectivity are required for P4.2.
