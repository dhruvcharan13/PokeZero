# What was built, and how the simulator works

Plain-language walkthrough. Facts about the engine come from experiments I ran; guesses are marked.

## 1. How the simulator works

**What it is.** The game engine is not Python. It is a compiled library (`libcg.so`) bundled inside the
`kaggle-environments` pip package (`envs/cabt`, "Card Battle"). Python only calls into it through
`ctypes`. I never saw its source; everything below is from calling it and looking at what comes back.

**The game loop is "answer the current question".**
1. `BattleStart(deck0 + deck1)` takes two 60-card decks (lists of card ids) and returns a handle to a game.
2. `GetBattleData(handle)` returns the current situation as JSON:
   * `select`: the question being asked of one player: how many answers they must give (`minCount`..`maxCount`)
     and a list of `option`s (each a small dict like `{"type": 13, "attackId": 1046}` or
     `{"type": 8, "area": 2, "index": 1, "inPlayArea": 4, "inPlayIndex": 0}`).
   * `current`: the board: both players' active/bench Pokémon (HP, attached energy, tools), discard piles, deck/hand/prize
     *counts*, turn number, flags like "energy already attached this turn". Only the acting player's own hand is
     listed; the opponent's hand is `null` and prize cards are `null`.
   * `logs`: recent events. `search_begin_input`: an opaque text blob (see section 3).
3. `Select(handle, [indices])` answers the question with the indices of the options you pick. The engine then runs
   until it needs another decision, from either player, and you read the data again. When `current.result` is 0 or 1
   that seat won; otherwise a draw.

So an "agent" is just a function from that JSON to a list of indices. Almost every decision is a menu (play a card, attach
energy, attack, end turn...). Some are follow-up prompts in the middle of an effect (choose a card from your deck, etc.).

**Option types.** Observed type numbers are 0,1,2,3,6,7,8,9,10,12,13,14,15. I **inferred** that 13 = attack,
14 = end turn, 8 = put a hand card on a Pokémon (attach/evolve), 7 = other action. The SDK does not document them.

**Speed.** About 100 random-vs-random games per second on this machine; a game has ~77 decisions.

**Things that surprised me.**
* No seed. Same decks, same picks, different game each time. So a game can't be replayed by re-running it; I record the
  list of picks instead.
* Only one game can be "live" in the stock Python wrapper (it keeps the pointer in a global). I re-wrote the thin wrapper so
  several games can coexist.
* During turn-0 setup your own face-down Pokémon appear as `null`.

## 2. The search API (what MCTS would need)

The library also exports `SearchBegin` / `SearchStep` / `SearchEnd`. Give `SearchBegin` the `search_begin_input` from an
observation and it creates a copy of the game you can step forward with any picks, as many branches as you like
(I stepped one parent four times; all worked). That is the "clone the state" question from the brief: **yes, you can**.
Stepping is random: the same move from the same state can give different results (shuffles, coin flips), so the tree has to
treat outcomes as chance events.

**The problem:** `SearchBegin` accepts arguments for the opponent's hidden deck, prizes and hand, but ignores them. The
opponent's real hand showed up in the copied game in every test (3 of 3), even when I passed a wrong one. So search on this
engine can see the opponent's actual cards. That breaks requirement 4 ("never access hidden information"), so it is fenced off
(see below). I don't know whether the competition allows `Search*` in submitted agents; I couldn't read the rules.

## 3. What I built (all under `src/pokezero/`)

| Piece | What it does |
|---|---|
| `env/engine.py`, `env/cabt.py` | Talks to the library. `CabtEnv.reset()` / `.step(picks)` / `.reward(seat)`. Validates picks (right count, no duplicates, in range) before the engine sees them. Records each game as a list of picks. |
| `env/base.py` | Competition-neutral types (`Decision`, `Trajectory`). Everything else uses these, so the competition code is isolated. |
| `env/cards.py` | Loads the engine's card and attack tables (1,431 cards) for numeric features (HP, type, retreat cost, best attack damage...). |
| `agents/random_agent.py`, `agents/heuristic.py` | Baselines. Heuristic: play cards first, then attack with the highest damage, end turn last; falls back to random on any error. Beats random ~90%. Its notion of option types is my guess from watching play. |
| `eval/arena.py`, `eval/elo.py` | Plays matches with seats alternated, round-robin tables, win rate and Elo. |
| `model/encoding.py` | Turns a decision into numbers: board slots, own hand, discard piles, counters, plus one row per option. **Reads only what the acting player is allowed to see** (a test mutates the opponent's hidden zones and checks the encoding does not change). |
| `model/net.py` | PyTorch network, ~2.2M parameters: card embeddings, a slot encoder, a trunk, an action encoder that scores each option, a policy (softmax over legal options only) and a value (−1..1, "how likely am I to win"). Prompts that need several picks are answered one pick at a time, with a STOP choice. |
| `search/puct.py` | AlphaZero-style tree search: the network's probabilities steer which moves to explore, its value judges leaf positions, illegal moves are masked, random outcomes become separate child nodes (capped), and the tree is reused where the position matches exactly. Refuses to run unless `allow_engine_oracle=true`. |
| `agents/az.py` | `SearchAgent` (search; self-play only, uses the leaky engine) and `PolicyAgent` (network only; what you would submit). |
| `train/selfplay.py`, `replay.py`, `learner.py`, `loop.py` | The cycle: the champion network plays itself with search; each move is stored as (what it saw, how often search visited each option, who won); a bounded buffer keeps recent examples; the learner trains policy cross-entropy + value error + weight decay; the candidate then plays the champion and replaces it only if it wins at least a configurable fraction (default 55%). Checkpoints, replay data and metrics are saved. |
| `storage.py`, `tracking.py`, `config.py`, `cli.py` | Local-disk artifact store behind an interface (S3/GCS can slot in), JSONL metrics with optional off-by-default W&B, typed YAML config with `a.b=value` overrides, `pokezero train/eval/bench/package`. |
| `submission.py` | Builds `submission.tar.gz` with a policy-only `main.py`; no search in it. |
| `tests/`, `Dockerfile`, `.github/workflows/ci.yml`, `README.md`, `docs/sdk-findings.md` | 31 tests; Docker (not built here); CI; docs. |

## 4. What the results do and don't show

* Works: the full loop runs, files are written, resume works, the policy never outputs an illegal move, the submission archive plays legal games.
* Does not work yet as a *player*: after 3 tiny iterations the network just ends its turn every turn and loses to random. That is
  normal for from-scratch training with 16 search simulations and a handful of games; I have not shown learning of useful play.
* Training targets come from a search that can see the opponent's hand. A policy trained that way is learning from information
  it won't have in a real match.

## 5. Things I assumed that you may want to challenge

* Both players use one fixed starter deck, and search assumes the opponent has the same deck.
* Single battles, not best-of-3.
* Option and area meanings listed above are guesses from behaviour.
* The competition page/rules were not readable from my environment.
