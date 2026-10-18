# Competition SDK findings

Everything here was observed by running the engine that ships in `kaggle-environments==1.33.0`
(`kaggle_environments.envs.cabt`, native `libcg`) on 2026-10-17. Items marked **inferred** were
deduced from behaviour, not documented. Items under "Unresolved" were *not* verified.

The container used to write this could not reach `kaggle.com` (proxy 403); the Playground rules were
later pasted by the user (see below). The engine facts come from the PyPI package and from experiments.

## Confirmed

| Topic | Finding |
|---|---|
| Package | The environment is `kaggle_environments.envs.cabt` ("Card Battle"). `cabt.json`: 2 agents, reward `-1/0/1`, default `bo=3`, `actTimeout=0`, `runTimeout=2000`, `remainingOverageTime=600`. |
| Agent contract | `agent(obs) -> list[int]`. When `obs["select"] is None` the agent returns its **60-card deck** (list of card ids). Otherwise it returns indices into `obs["select"]["option"]`, between `minCount` and `maxCount` of them (the starter agents use `random.sample(range(len(option)), maxCount)` / `range(maxCount)`). |
| Observation | `select` (`type`, `context`, `minCount`, `maxCount`, `option[]`, `deck`, `contextCard`, `effect`, ...), `current` (public board; `players[i].hand` is a list for the acting player and `null` for the opponent; `prize` entries are `null`), `logs` (typed event list), `search_begin_input` (opaque ASCII payload). |
| Action space | Variable-size, structured: each option is a dict with a `type` (observed 0,1,2,3,6,7,8,9,10,12,13,14,15) and type-specific fields (`area`, `index`, `playerIndex`, `inPlayArea`, `inPlayIndex`, `attackId`, `cardId`, `count`, `energyIndex`). Multi-pick prompts (`maxCount` up to 3 observed) take a list. `maxCount` can exceed the number of options, so picks must be capped. |
| Prompt kinds | 9 distinct `(select.type, min, max)` combinations seen over random play; type 0 is the main-phase menu (≈60% of decisions). `select.context` (12 values seen) says why the prompt exists. |
| Option types | Official enum (matsuoinstitute.github.io/cabt): 0 NUMBER, 1 YES, 2 NO, 3 CARD, 4 TOOL_CARD, 5 ENERGY_CARD, 6 ENERGY, 7 PLAY, 8 ATTACH, 9 EVOLVE, 10 ABILITY, 11 DISCARD, 12 RETREAT, 13 ATTACK, 14 END, 15 SKILL, 16 SPECIAL_CONDITION. Select types 0-10 (MAIN, CARD, ATTACHED_CARD, ..., YES_NO, SPECIAL_CONDITION); `context` has 49 documented values (e.g. 0 MAIN, 41 IS_FIRST, 46 COIN_HEAD). My first guesses were partly wrong (e.g. I had 7/8 as one family and no RETREAT); the heuristic and docs now follow the official enum. |
| Areas | Official: 1 DECK, 2 HAND, 3 DISCARD, 4 ACTIVE, 5 BENCH, 6 PRIZE, 7 STADIUM, 8 ENERGY, 9 TOOL, 10 PRE_EVOLUTION, 11 PLAYER, 12 LOOKING. (I had guessed 6 = deck; it is the prize area.) The encoder only resolves hand/active/bench references today. |
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

## Playground rules and format (pasted from the competition page on 2026-10-18)

My own fetches of kaggle.com only return a title (the tabs are JavaScript-rendered), so the user pasted the
Playground's Data/Rules/Overview text. Key facts:

* **The Playground is open**: started Sept 29 2026; entry and team-merger deadline Jan 1 2027; final submission
  deadline Jan 8 2027; evaluation continues to Jan 22 2027. It is a continuation of the closed Simulation
  competition (that one ended Aug 17; the Strategy report track ended Sept 14). Prizes are Kaggle swag only.
* **Format**: a Kaggle Environments simulation; submissions play a ladder against similar-rated agents
  (Gaussian skill rating, initial mu = 600, latest **2** submissions active, **5 submissions/day**), then a
  Bradley-Terry tournament at the end. The brief says "best of three" for the research aim; the ladder
  episodes may be bo3 (the local environment's default `bo` is 3).
* **Submission**: `.tar.gz` with `main.py` at the top level (not nested) and a `deck.csv` (60 card ids, one
  per line). At most **197.7 MiB**. Files are placed in `/kaggle_simulations/agent/`. A validation game
  against itself runs first. Resources: 2 vCPU, 12.2 GiB RAM, 11.8 GiB disk. The grader image is built on
  Kaggle's Python image (`gcr.io/kaggle-images/python`, so numpy is certain and torch is likely but
  **not confirmed**) with `kaggle-environments`; the engine comes from that package, so shipping the native
  `cg/` folder is not required by this page.
* **Rules that matter here**: no ingress/egress during an episode (no network from the agent); external data
  and models are allowed if reasonably accessible; no private code sharing outside a team (public sharing on
  Kaggle is fine); **you may not publicly share Pokemon Elements** (card names/text, card faces, game logic data,
  deck recipes) or use models trained on them outside the competition; Competition Data is "competition use
  only" (do not redistribute the card CSV/PDF); episode replays may be public and downloadable.
* **Docs** (official, https://matsuoinstitute.github.io/cabt/): the observation schema, the enums above,
  `all_card_data()`, `all_attack()`, `deck.csv` format, and a `search_begin / search_step / search_end /
  search_release` API listed by name only: **no argument list, hidden-information semantics or lookahead
  restriction is documented on the pages I could read**. The competition rules text I was given does not forbid
  search.
* Episode replays from the leaderboard and a daily export of top episodes (for behaviour cloning / RL / IL)
  are promised in the forums; the CLI docs are at `Kaggle/kaggle-cli/docs/simulation_competitions.md`.
* The official site (not the Playground) also states 10 minutes per player per match, timeout = loss; the
  Playground text does not repeat a time limit, so treat 10 minutes per match as the safe assumption.

Community repos (read-only, unverified) add: ISMCTS-style determinized search lost on the ladder in one repo
even after winning locally; two repos saw no self-play learning gain.

### Consequence for this repo

`pokezero package` now builds the documented layout (`main.py`, `deck.csv`, `pokezero/`, `model.pt`) and
checks the size cap. Remaining risks: PyTorch in the grader is unconfirmed (the agent falls back to the heuristic
if the network cannot load), `main.py` has never run on the real ladder, and our package imports
`kaggle_environments` (present in the grader image per the Docker reference).

## Kaggle's replay viewer (confirmed)

`kaggle_environments.make("cabt")` + `env.run([...])` + `env.render(mode="html")` produces a ~2 MB
self-contained page (`envs/cabt/visualizer/default/dist/index.html`, 612 KB) with a step slider. Its data
is the runner's `visualize` payload, which is **omniscient**: both players' hands and prize ids.
That is fine for spectating and does not change agents' observations. `pokezero replay` feeds our agents through
this runner. `CabtEnv` does not use the runner for training because the runner stores one battle in a
module global, plays `bo` battles, and is slower; this has nothing to do with the search leak.

## What other entries report

(Superseded in part by the Playground rules above: the grader image is Kaggle's Python image, so numpy is available and torch is likely.)

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

1. **Search semantics**: the official docs list `search_begin` etc. without arguments. Is passing predicted hidden
   information supposed to work (a leak-free lookahead), and is my observation that the arguments are ignored
   a bug or intended? Is search allowed in submissions? (The pasted rules do not forbid it.)
2. Is PyTorch available in the grader image? (Likely, from the base image; unconfirmed.) Per-move and per-match time
   limits for the Playground (10 min/match on the official site).
3. Does the ladder play bo3 with the same deck? We train on single battles with one fixed deck and never build decks.
4. Whether `search_begin_input` can be turned into a belief-sampled hypothetical *without* revealing the
   true state (no such API is documented).
