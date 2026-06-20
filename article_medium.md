# Clustering 40M Unstructured Logs a Day on a Laptop with (almost) Zero Regex

*A lightweight explainer on two elegant ideas — and a tool that uses both*

---

## The Problem with Logs

You just shipped a deploy. Everything looks green — dashboards nominal, no alerts firing. But did it *actually* go clean?

You have 40M log lines a day flowing into Elasticsearch. Somewhere in there, the deploy may have introduced new error patterns you've never seen before. The question isn't "are there errors?" — there are always errors. The question is: **did anything new show up after this deploy that wasn't there before?**

Here's what a typical minute of WARNING/ERROR logs looks like in a trading system:

```
WARNING  order a3f9b12c rejected: price 184.52 outside allowed slippage for SPY
WARNING  order 77d401ea rejected: price 184.61 outside allowed slippage for SPY
WARNING  order c90e3311 rejected: price 184.49 outside allowed slippage for SPY
ERROR    market data feed 9f2a timeout after 2003ms, symbol=AAPL session=4d71c8
ERROR    market data feed b31e timeout after 1987ms, symbol=AAPL session=3a09f1
WARNING  position limit breached: account f3a09b21 held 14823 shares of IWM, limit=14000
WARNING  position limit breached: account 2c18d77f held 14951 shares of IWM, limit=14000
ERROR    risk check failed for order 8e1cd920: margin utilization 94.3% exceeds threshold
ERROR    risk check failed for order 551fa7b3: margin utilization 91.7% exceeds threshold
```

That's one minute. You have 1,440 minutes in a day, 15,000+ instruments, dozens of services. You can't eyeball it. You can write Kibana queries, but only for errors you already know to look for.

What you really need is a diff — *yesterday's log patterns* vs *today's log patterns* — so anything new jumps out. And the keyword is *patterns*: you don't want to compare individual lines (every order ID is unique), you want to compare the *shapes* of log lines across days.

Two mathematical ideas — Markov chains and Jaccard similarity — make that diff possible. Let me explain both, simply, then show how they combine.

---

## Part 1: Markov Chains — Does This Token Look Like English?

Imagine you run a diner. You track what customers tend to order together. Over time, you notice:

- Someone who orders **bacon** has a 70% chance of also ordering **eggs**, and a 40% chance of ordering a **banana shake**
- Someone who orders **hamburger** has a 5% chance of ordering a **banana shake**, but an 80% chance of ordering **fries**
- Someone who just ordered **banana** ... probably isn't ordering a hamburger next

You've built a **Markov chain** — a model where the probability of the *next* item depends on the *current* item. Each state (menu item) has a table of transition probabilities to the next state.

Now flip this around: if someone orders `xq7f29m`, you can ask your chain — *how likely is this sequence, given what I know about how real orders flow?* The answer: essentially zero. `xq7f29m` is not on any menu. It doesn't follow any known pattern. **It's probably noise.**

This is exactly how `fuzzylogs` uses Markov chains — but on *characters* instead of menu items. The model is trained on an English dictionary and learns, for each pair of characters, how likely the next character is. Real words like `failed` or `login` produce high probabilities. A SHA hash like `8f3a9c21` produces near-zero probability. The token gets replaced with `.`.

The chain doesn't need to know what a UUID is. It just knows `8f3a9c21` doesn't *feel* like English — and that's enough.

---

## Part 2: Jaccard Similarity — How Alike Are Two Sets?

You have two bags of groceries.

**Bag A:** `{eggs, milk, bread, butter}`  
**Bag B:** `{eggs, milk, cheese, yogurt}`

How similar are they? Jaccard similarity answers this precisely:

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
             = |{eggs, milk}| / |{eggs, milk, bread, butter, cheese, yogurt}|
             = 2 / 6
             ≈ 0.33
```

Items in common, divided by items total. The result is always between 0 (nothing in common) and 1 (identical).

Now apply this to log lines:

**Line A (fuzzed):** `ERROR user . failed login from .`  
**Line B (fuzzed):** `ERROR user . failed login from .`

These are identical sets of tokens — Jaccard similarity = 1.0. Same cluster.

**Line C (fuzzed):** `WARN session . expired after . seconds`

Jaccard(A, C) ≈ 0.1. Different cluster.

No regex required. No schema. Just set math.

---

## Part 3: Putting It Together — fuzzylogs

[fuzzylogs](https://github.com/larytet-py/fuzzylogs) is a small Python tool that chains these two ideas into a two-pass pipeline:

**Pass 1 — Fuzz:** Run each token through the Markov chain. If it doesn't look like an English word, replace it with `.`. Your 40,000 log lines collapse into a much smaller set of *patterns*.

**Pass 2 — Cluster:** Compare fuzzed lines using Jaccard similarity. Lines above a similarity threshold get grouped together. Count the members of each group.

The output isn't a wall of log lines. It's a ranked summary:

```
38,947x  ERROR user . failed login from .
   821x  WARN  session . expired after . seconds
    12x  ERROR database . connection timeout
```

Now you know: this is one problem, happening 38,947 times. The other two patterns are noise by comparison. You can sleep (or escalate, with actual information).

---

## Why This Combination Works

The Markov chain handles *vocabulary* — separating meaningful words from dynamic values without any hardcoded rules for what counts as an ID, a hash, or an IP. It generalizes.

Jaccard handles *structure* — grouping lines that share the same template without caring about word order edge cases or requiring exact string matches. It's forgiving.

Neither technique alone is enough. Jaccard on raw log lines would treat `user 8f3a9c21` and `user b2d77f03` as different. The Markov fuzzing step is what makes them look the same before Jaccard does its work.

Together, they turn log archaeology into something closer to triage.

---

*The code is ~200 lines of Python, no dependencies beyond a system dictionary. Worth a read if you like elegant small tools: [github.com/larytet-py/fuzzylogs](https://github.com/larytet-py/fuzzylogs)*
