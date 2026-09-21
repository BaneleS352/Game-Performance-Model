# Persistence movement decision - v14

Acceleration remains 5%. Current Performance remains 65%, Momentum 15%, Acceleration 5%, Stability 15%.

Persistence is a movement multiplier. It is not a ranking-score component and it is not a fixed waiting period.

The selected movement policy is evidence-targeted: signal strength from evidence, momentum and acceleration creates the base movement pressure; persistence increases that pressure only when the underlying evidence is meaningful. Established strong games retain protection against abrupt downward replacement.

Real-data test on the existing 30 evaluation days and 3,326 games: the targeted multiplier produced approximately 0.682 rank correlation, 51.7% Top-10 overlap, 54.0% Top-20 overlap, 76.2% Top-100 overlap, 377.2 mean rank error, 7.0 average absolute daily movement, 5 median movement, 36 maximum movement, 5.7% of moves above 20 positions, and 24.7% direction reversals.

The previous v13 baseline was approximately 0.679 correlation, 52.3% Top-10, 55.0% Top-20, 76.3% Top-100, 380.1 rank error, 6.1 average movement, 32 maximum movement, 2.3% moves above 20, and 24.9% reversals.

The trade-off is intentional: movement becomes more responsive for persistent evidence, while broad ranking quality remains close to baseline. High-evidence games with 4+ days of persistence moved about 12.8 positions on average in the test, which demonstrates that the feature is not limited to one-position-per-day behaviour.
