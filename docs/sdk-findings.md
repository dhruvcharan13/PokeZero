# Competition SDK findings

Everything here was observed by running the engine that ships in `kaggle-environments==1.33.0`
(`kaggle_environments.envs.cabt`, native `libcg`) on 2026-10-17. Items marked **inferred** were
deduced from behaviour, not documented. Items under "Unresolved" were *not* verified.

The container used to write this could not reach `kaggle.com` (proxy 403), so competition
**rules, data files and the official starter notebook were not read**. Everything below comes from
the PyPI package and from experiments.

## Confirmed

| Topic | Finding |
|---|---|
| Package | The environment is `kaggle_environments.envs.cabt` ("Card Battle"). `cabt.json`: 2 agents, reward `-1/0/1`, default `bo=3`, `actTimeout=0`, `runTimeout=2000`, `remainingOverageTime=600`. |
| Agent contract | `agent(obs) -> list[int]`. When `obs["select"] is None` the agent returns its **60-card deck** (list of card ids). Otherwise it returns indices into `obs["select"]["option"]`, between `minCount` and `maxCount` of them (the starter agents use `random.sample(range(len(option)), maxCount)` / `range(maxCount)`). |
| Observation | `select` (`type`, `context`, `minCount`, `maxCount`, `option[]`, `deck`, `contextCard`, `effect`, ...), `current` (public board; `players[i].hand` is a list for the acting player and `null` for the opponent; `prize` entries are `null`), `logs` (typed event list), `search_begin_input` (opaque ASCII payload). |
| Action space | Variable-size, structured: each option is a dict with a `type` (observed 0,1,2,3,6,7,8,9,10,12,13,14,15) and type-specific fields (`area`, `index`, `playerIndex`, `inPlayArea`, `inPlayIndex`, `attackId`, `cardId`, `count`, `energyIndex`). Multi-pick prompts (`maxCount` up to 3 observed) take a list. `maxCount` can exceed the number of options, so picks must be capped. |
| Prompt kinds | 9 distinct `(select.type, min, max)` combinations seen over random play; type 0 is the main-phase menu (≈60% of decisions). `select.context` (12 values seen) says why the prompt exists. |
| Option types (**inferred**) | 13 = attack (`attackId`), 14 = end turn, 8 = play a hand card onto a Pokemon (attach/evolve), 7 = other hand/in-play action, 3/6/15 = card/energy picks inside effect prompts, 1/2 = coin-style yes/no at setup, 0 = numeric choice. |
| Areas (**inferred**) | `area` 2 = hand (index into the acting player's hand), 4 = active, 5 = bench, 6 = deck (from logs). Other codes are not decoded. |
| Local simulator | `cg.game.battle_start/battle_select/battle_finish` wrap `libcg.so` via ctypes. Linux x86-64/arm64, macOS and Windows binaries are bundled. `Select` returns non-zero for illegal input. |
| One live battle per module | `cg.game` stores the battle pointer in a class attribute. `pokezero.env.engine` re-binds the same C functions and keeps the pointer per object, so several games can coexist (tested). |
| Speed (4-core CPU, random play) | ~85 games/s, ~77 decisions/game, ~6,300 decisions/s. |
| Deck validation | `BattleStart` takes `deck0 + deck1` (120 ints) and reports `errorPlayer/errorType`. The starter `cabt.deck` is accepted. |
| Card database | `AllCard()` / `AllAttack()` export JSON for 1,431 cards (type, HP, retreat, energy, evolution flags, attack ids) and all attacks (damage, energy cost, text). Used for card features. |
| Face-down setup | During turn-0 setup a player's own Pokemon appear as `null` entries in `active`/`bench`. |
| Game result | `current.result`: `-1` running, `0`/`1` winner seat, otherwise draw. Logs end with `{"type": 23, "result", "reason"}`. |

## Seeding and reproducibility

* `BattleStart` has **no seed parameter**. Calling `libc.srand` first changes nothing, and two
  games from the same deck with identical pick sequences diverge (observed: 140 vs 80 steps).
* Consequence: a game cannot be *re-simulated* from `(decks, seed)`. `pokezero` records every
  game as a `Trajectory` (decks, per-step player/picks/option count) and seeds only its own Python
  RNG (`reset(seed)`); tests assert agent determinism and trajectory serialization, not engine
  determinism.

## Cloning / search (the key question for MCTS)

The library exports `AgentStart`, `SearchBegin`, `SearchStep`, `SearchEnd`, `SearchRelease`
(no ctypes signatures upstream; declared in `pokezero/env/engine.py` following the community
`4dalols/pokemon-agent` repo).

* **State can be cloned for search**: `SearchBegin(agent, search_begin_input, len, myDeck, myPrize,
  oppDeck, oppPrize, oppHand, extra, flag)` returns `{"state": {"observation", "searchId"}, "error"}`.
  `SearchStep(agent, searchId, picks, n)` returns the successor. Stepping the same `searchId`
  repeatedly is non-destructive (4 branches from one parent all worked) and children can be stepped
  further, so a real tree is possible. `SearchEnd` frees every id.
* **Stepping is stochastic.** Chance (draws, shuffles, coin flips, opponent choices) is re-sampled
  on each `SearchStep`; the same parent yields different children. The tree must model this
  (`pokezero.search.puct`: outcome-keyed children per edge).
* **The simulator leaks hidden information (critical).** The opponent hidden-zone arguments
  (`oppDeck`, `oppPrize`, `oppHand`) are ignored: the opponent's *real* hand reappears after
  `SearchStep` in 3/3 experiments no matter what is passed (including a deliberately wrong hand).
  `search_begin_input` evidently encodes the full state. Search through it is therefore an
  **oracle**. `pokezero` isolates it behind `allow_engine_oracle` and the policy-only agent never
  touches it.
* `SearchBegin` returns `error=1` during the turn-0 setup phase (before prizes are dealt); the
  search agent falls back to the network prior there.
* Search throughput on 4 CPU cores with the 2.2M-parameter network: ~400-480 simulations/s
  (about 1 engine call per simulation).

## Kaggle's replay viewer (confirmed)

`kaggle_environments.make("cabt")` + `env.run([...])` + `env.render(mode="html")` produces a ~2 MB
self-contained page (`envs/cabt/visualizer/default/dist/index.html`, 612 KB) with a step slider. Its data
is the runner's `visualize` payload, which is **omniscient**: both players' hands and prize ids.
That is fine for spectating and does not change agents' observations. `pokezero replay` feeds our agents through
this runner. `CabtEnv` does not use the runner for training because the runner stores one battle in a
module global, plays `bo` battles, and is slower; this has nothing to do with the search leak.

## What other entries report

Taken from public repos/notes (I could not open Kaggle writeups; none of this was re-run by me):

* Most top-visible approaches are rule-based, behaviour-cloning or gradient-boosted rankers; search is
  usually a gated "override" on main-phase decisions, not a full tree.
* `4dalols/pokemon-agent`: determinized rollouts through `SearchBegin/Step`, ~68% vs a heuristic opponent
  (not independently checked). `Jun-Morita/kaggle-ptcg-ai-battle`: search with determinization; going
  from 1 to 4 determinizations was flat, and giving search the *correct* opponent deck gained +0.011
  (noise). Both are what you would see if the hidden-info arguments were ignored, but nobody
  I found states that.
* None of the pages I read mentions the leak. The README of one calls the API a way to give the
  opponent's hidden information "as predicted values", so the intended contract may differ from what I observed
  (unresolved).
* A from-scratch AlphaZero iteration in one repo collapsed to a 0.195 win rate because narrow visit targets
  (2-3 visits per candidate) flattened the policy: the same failure mode our demo showed. Their
  conclusion was that without a working search operator there is no policy-improvement step.
* Sandbox constraints reported by one competitor (unverified): **no numpy**, a limited archive size, ~1.6 vCPU, and
  a 600 s total budget per match with no per-move cap. Our `torch`/`numpy` submission would likely fail
  there; a pure-Python or exported-weights inference path would be needed.

## Unresolved questions

1. **Competition rules** (kaggle.com unreachable): is use of `Search*` inside a submitted agent
   allowed, and does the grader's `search_begin_input` carry the same full state? If search
   is allowed *with* leakage, the oracle is a competition feature and the "research-only" label can
   be dropped; if not, a leak-free forward model is required.
2. Does the submission runtime provide **PyTorch** (and numpy: see above), and what are its CPU/time limits per move
   (`actTimeout` is `0` here)? `pokezero package` falls back to the heuristic agent if torch is missing.
3. Deck rules for the real competition (the local engine only checked size 60 and card validity).
   A single fixed starter deck is used everywhere; deck building is out of scope.
4. Decoding of all `area` codes and option types 0, 1, 2, 9, 10, 12 (only used as opaque features).
5. Best-of-3 handling: the kaggle wrapper runs `bo` battles; this repo trains on single battles.
6. Whether `search_begin_input` can be turned into a belief-sampled hypothetical *without* revealing
   the true state (would need a custom state reconstruction; no such API is exported).
