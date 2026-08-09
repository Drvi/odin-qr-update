# 004 — Parallel accumulation by merging triangles

**The mechanism now exists.** `ols_accum_merge` was added for data
experimentation (combining separately collected segments), and row-parallel
OLS is the same operation: partition the rows, accumulate each partition into
its own triangle with no shared state and no locks, then merge pairwise.

Measured merge cost is 0.4-2.3 us for p = 8..24, against O(m*p^2) for the
accumulation itself, so the merge is free at any m worth parallelising.

Still not done: the threading itself. No measurement has been taken showing a
single thread is insufficient for a real workload, and the survey's modal
machine has 6-8 cores that a game is already using for other things. Build it
when a measurement demands it, not before.
