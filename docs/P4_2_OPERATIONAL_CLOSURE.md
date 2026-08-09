# P4.2 Operational Closure — IBKR Reference Cost Profile

**Project:** Smart Investment Bot
**Phase:** P4.2 — IBKR reference cost profile
**Closure date:** 9 August 2026
**Product mode:** PAPER_ONLY
**Decision:** COMPLETE

## 1. Executive conclusion

P4.2 is complete.

The paper-trading runtime now models the confirmed IBKR FIXED pricing plan without connecting to an IBKR account. The reference-cost model is versioned, validated and wired into both candidate economics and the paper order lifecycle.

The cost gate rejects trades that cease to meet the required reward-to-risk threshold after modeled transaction costs.

No IBKR credentials, broker API connection or live-execution capability were introduced.

## 2. Confirmed IBKR operating assumptions

The active reference profile is:

- Provider: IBKR
- Profile: `config/ibkr_reference_costs_v2.json`
- Profile version: `ibkr-reference-2026-08-09-v2`
- Active pricing plan: `FIXED`
- Intended routing assumption: `IBKR_SMARTROUTING`
- API connection: disabled
- Active per-trade FX mode: none
- Entry FX conversion per stock trade: disabled
- Exit FX conversion per stock trade: disabled
- FX funding control: `MANUAL_PORTFOLIO_FUNDING_EVENT`
- USD sale-proceeds policy: `RETAIN_USD`

The confirmed FIXED pricing-plan assumption is based on user-confirmed account transaction history together with the official IBKR pricing schedule.

The historical schema-v1 reference profile remains preserved and inactive.

## 3. Versioned cost model

The P4.2 reference profile contains:

- US FIXED stock/ETF commission rules
- US Tiered reference rules
- minimum commission behavior
- maximum commission caps
- fractional-share commission and minimum rules
- applicable US regulatory-cost components
- European EUR reference schedules
- spot-FX reference costs
- automatic-conversion reference costs

European EUR execution remains explicitly `REFERENCE_ONLY_MARKET_SPECIFIC`.

The current operational universe is USD-quoted US equities. Non-USD lifecycle execution remains fail-closed until an approved venue-specific cost model is available.

Fractional-share rules are modeled in the profile, but the current execution engine continues to use whole-share sizing.

## 4. Cost-adjusted trade economics

P4.2 calculates transaction economics before an order is created.

The model evaluates:

- entry commission
- exit commission
- applicable regulatory costs
- optional FX conversion costs when explicitly enabled
- round-trip cost
- gross reward-to-risk
- cost-adjusted net reward-to-risk

A deterministic hand-calculated example produced:

- Entry notional: USD 110
- Portfolio FX rate: 0.90
- Portfolio notional: EUR 99
- Gross reward-to-risk: 3.0
- Reward-path modeled cost: EUR 1.86819021
- Risk-path modeled cost: EUR 1.86184233
- Net reward-to-risk: 1.90577074

The example was therefore correctly classified as uneconomic against the required reward-to-risk threshold.

## 5. Authoritative lifecycle fees

The paper lifecycle now routes broker-cost estimation through one authoritative IBKR lifecycle estimator whenever an IBKR pricing plan is selected.

The estimator is used for:

- legacy quantity affordability
- fixed-notional quantity sizing
- order reservation
- entry fill
- position close

The generic execution-cost model remains only as the fail-safe fallback when no IBKR pricing plan is selected.

For incomplete Tiered estimates, sizing may use the known cost portion so the economic cost gate can own the final rejection. Reservation, fill and close remain fail-closed when an authoritative estimate is incomplete.

## 6. Runtime activation

Product policy version `p4.2-3` activates:

- `ibkr_cost_gate_enabled = true`
- `ibkr_pricing_plan = FIXED`

The production `build_runtime()` path now loads the validated product policy and explicitly constructs `AutomatedExecutionConfig` from it.

Fail-safe defaults remain inactive when `AutomatedExecutionConfig()` is constructed directly without the validated runtime-policy loader.

The runtime activation does not enable broker connectivity.

The following remain false:

- IBKR API connection
- broker API connection
- live trading
- per-trade entry FX conversion
- per-trade exit FX conversion

## 7. EUR 2,000 / USD-quoted operational acceptance proof

The final P4.2 gate used temporary SQLite databases only and reproduced the P4 operational shape:

- Portfolio currency: EUR
- Starting balance: EUR 2,000
- Security quote currency: USD
- Static acceptance-test FX rate: USD/EUR 0.90
- Pricing plan: IBKR FIXED
- Cost gate: enabled

Accepted candidate:

- Entry price: USD 50
- Quantity: 2 shares
- USD notional: USD 100
- EUR notional: EUR 90
- Reserved cash: EUR 90.90000540
- Planned loss: EUR 5.40
- Entry IBKR reference fee: USD 1.00000600
- Exit IBKR reference fee: USD 1.00307400
- Booked lifecycle fees: EUR 1.80277200
- Final net P&L: EUR 25.19722800
- Account reconciliation: true

The order remained below the EUR 100 hard ceiling and below the EUR 10 planned-loss cap.

An intentionally uneconomic candidate produced:

- Created orders: 0
- Rejected entries: 1
- Error type: `IBKRCostGateRejected`
- Gross reward-to-risk: 2.00000000
- Net reward-to-risk: 1.15036698
- Minimum required reward-to-risk: 2.00000000
- Cost estimate complete: true

The rejection occurred before an order was created.

## 8. Operational database preservation

The operational P4 account was not used for the final acceptance proof.

Operational account:

`ACC-749ca5703d214ef0b91f87b825e88849`

Accepted operational database SHA256 before and after all P4.2 implementation and proof work:

`45364e65e2f95bd4b016c8332ed37410fe8ed494d8ce9917546e21792401a22e`

The activation proof used temporary databases and did not alter the operational portfolio.

Historical P3 evidence remains preserved separately and was not reconstructed into the P4 operational database.

## 9. Implementation checkpoints

P4.2 was delivered incrementally through these pushed checkpoints:

- `68b5236` — IBKR reference-cost core
- `e1ae0df` — cost-adjusted long-trade economics
- `1dd289b` — IBKR cost gate wired into execution
- `2dd53f9` — P4.2 profile policy pinned
- `0095f7b` — verified IBKR reference profile v2
- `7bc7597` — paper lifecycle routed through IBKR reference fees
- `b0c762d` — confirmed FIXED cost gate activated in runtime

The final local regression after activation passed:

`488 passed`

No GitHub Actions result for the exact final P4.2 activation checkpoint is claimed in this closure record.

## 10. Manual maintenance workflow

The manual maintenance and pricing-change workflow remains documented in:

`docs/P4_2_IBKR_COST_PROFILE.md`

Any future pricing change must continue to require:

- a new versioned reference profile
- official IBKR source verification
- hand-calculated examples
- regression testing
- explicit pricing-plan and FX-mode review
- no credentials in the repository
- no broker connectivity unless approved by a future phase

## 11. P4.2 closure gate

- [x] Versioned IBKR commission profile exists.
- [x] Confirmed FIXED pricing plan is represented.
- [x] Minimum commission rules are modeled.
- [x] Regulatory-cost components are modeled.
- [x] Fractional-share and minimum rules are modeled.
- [x] FX cost models are represented.
- [x] Manual FX workflow matches the operational account workflow.
- [x] Round-trip transaction costs are calculated.
- [x] Net reward-to-risk is calculated after costs.
- [x] Hand-calculated economics match deterministic implementation output.
- [x] Uneconomic EUR 100-scale trades are rejected before order creation.
- [x] IBKR reference fees are used by reservation, fill and close lifecycle paths.
- [x] EUR 2,000 / USD-quoted lifecycle remains inside the EUR 100 order ceiling.
- [x] Planned loss remains inside the EUR 10 cap.
- [x] Cross-currency lifecycle accounting reconciles.
- [x] Runtime loads the validated active cost policy.
- [x] Non-USD execution remains fail-closed.
- [x] Full regression passes with 488 tests.
- [x] Operational P4 database remains unchanged.
- [x] No IBKR credentials are required.
- [x] No IBKR API or broker connectivity exists.
- [x] Live execution remains disabled.

**Final decision: P4.2 COMPLETE.**

## 12. Controlled transition to P4.3

P4.3 may now begin.

The next phase separates swing and medium-term strategy behavior while preserving all P4.0-P4.2 safety invariants.

P4.3 must not weaken:

1. Paper-only execution.
2. EUR portfolio sizing controls.
3. The EUR 100 hard order ceiling.
4. The EUR 10 planned-loss cap.
5. IBKR cost-adjusted economic acceptance.
6. Manual FX assumptions.
7. Broker/API disconnection.
8. Reconciliation and fail-closed behavior.

P4.4 and later phases remain not started.
