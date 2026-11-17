# Audit of extreme daily returns

Every daily log return with |r| > 0.20 is flagged by `scripts/02_build_network.py`
(NSE's widest daily price band for ordinary stocks is 20%, so anything beyond it
needs an explanation). The full list is written to
`results/tables/data_extreme_return_audit.csv`. Each flagged return was
inspected against the unadjusted close and the traded volume.

Two kinds of *mechanical* price resets were found. They are not economic
returns, so the return on that single day is set to zero for that single stock
(`preprocess.return_overrides` in `config.yaml`):

1. **Corporate actions that Yahoo did not back-adjust across a calendar-year
   boundary.** The price series steps down on 1 January by exactly the
   split/bonus factor, on ordinary volume, although the action took place months
   later (GPIL, CGCL, MOTILALOFS on 2024-01-01; CONCOR on 2025-01-01; TRENT on
   2026-01-01).
2. **Demerger ex-dates.** The share price falls because part of the business
   leaves the listed entity, not because shareholders lost that value
   (TMPV 2025-10-14, STLTECH 2025-04-24, VEDL 2026-04-30, HEGAM 2026-09-07).

All other flagged returns are genuine market events and are kept. The largest
coincide with a jump in traded volume to 5-40 times the 20-day median and with
identifiable news; about a third fall on the general-election result day,
2024-06-04, when many stocks closed at or near the 20% price band.

`psu_list.csv` in this folder is a hand-compiled list of Nifty 500 companies in
which the central or a state government (directly or through another
government-owned company) holds a majority. It is used only to colour and
group nodes; it does not enter the construction of the network.
