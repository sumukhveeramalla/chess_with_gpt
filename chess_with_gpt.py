import os
import re
import json
import random
import streamlit as st

# Rules & board rendering
import chess
import chess.svg

# OpenAI client (optional)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # we'll handle this gracefully


# =========================
# Helpers
# =========================
MOVE_PHRASE = re.compile(
    r"(?P<from>[a-h][1-8])\s*(to|\-|\s)\s*(?P<to>[a-h][1-8])\s*(?P<promo>=?[qrbnQRBN])?$"
)
CENTER = {chess.D4, chess.E4, chess.D5, chess.E5}


def parse_human_move(text: str, board: chess.Board):
    """Parse 'e2e4', 'b1 to d3', 'Nf3', 'O-O', etc., and return a legal chess.Move or None."""
    t = text.strip()

    # Try UCI first (e2e4, e7e8q)
    try:
        uci = t.lower().replace(" ", "").replace("=", "")
        mv = chess.Move.from_uci(uci)
        if mv in board.legal_moves:
            return mv
    except Exception:
        pass

    # Try "b1 to d3"
    m = MOVE_PHRASE.search(t.lower())
    if m:
        uci = f"{m['from']}{m['to']}"
        promo = m.group("promo")
        if promo:
            uci += promo.lower().replace("=", "")
        try:
            mv = chess.Move.from_uci(uci)
            if mv in board.legal_moves:
                return mv
        except Exception:
            pass

    # Try SAN (Nf3, O-O, exd5, e8=Q)
    try:
        mv = board.parse_san(t)
        if mv in board.legal_moves:
            return mv
    except Exception:
        pass

    return None


def unicode_board(board: chess.Board) -> str:
    return board.unicode(borders=True, invert_color=False)


def svg_board(board: chess.Board) -> str:
    last = board.move_stack[-1] if board.move_stack else None
    return chess.svg.board(
        board=board,
        coordinates=True,
        lastmove=last,
        check=board.king(board.turn) if board.is_check() else None,
    )


def create_client(api_key: str | None):
    if not OpenAI or not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def json_lenient_loads(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        lines = [ln for ln in s.splitlines() if not ln.strip().lower().startswith("json")]
        s = "\n".join(lines)
    return json.loads(s)


def assistant_move_with_model(client, board: chess.Board, model: str = "gpt-4o-mini"):
    """Choose a move only from explicit legal UCIs. Raises if output isn't legal."""
    legal_uci = sorted(m.uci() for m in board.legal_moves)
    if not legal_uci:
        raise RuntimeError("No legal moves (game over).")

    system = (
        "You are playing a legal chess game as the ASSISTANT.\n"
        "Choose exactly ONE move from the provided legal_uci list.\n"
        'Respond ONLY as JSON: {"uci": "<one string from legal_uci>", "explanation": "..."}.\n'
        "Do not include any other keys or text."
    )
    user = f"FEN: {board.fen()}\nlegal_uci: {legal_uci}"

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0,  # make it obedient
    )
    data = json_lenient_loads(resp.choices[0].message.content)
    uci = data.get("uci")
    if not uci or uci not in legal_uci:
        raise RuntimeError(f"Model chose invalid move: {uci!r}")
    mv = chess.Move.from_uci(uci)
    return mv, data.get("explanation", "")


def show_game_end(board: chess.Board):
    if board.is_checkmate():
        winner = "Black" if board.turn else "White"
        st.success(f"🏁 Checkmate! {winner} wins.")
    elif board.is_stalemate():
        st.info("🏁 Stalemate.")
    elif board.is_insufficient_material():
        st.info("🏁 Draw by insufficient material.")
    elif board.can_claim_fifty_moves():
        st.info("🏁 Draw by 50-move rule (claimable).")
    elif board.can_claim_threefold_repetition():
        st.info("🏁 Draw by threefold repetition (claimable).")


def moves_table_markdown(board: chess.Board, start_fen: str) -> str:
    """Rebuild SAN history from the true start FEN into a markdown table."""
    tmp = chess.Board(start_fen)
    moves = list(board.move_stack)
    lines = ["| # | White | Black |", "|---:|:-----|:------|"]
    ply = 0
    move_num = 1
    while ply < len(moves):
        w = tmp.san(moves[ply]); tmp.push(moves[ply]); ply += 1
        b = ""
        if ply < len(moves):
            b = tmp.san(moves[ply]); tmp.push(moves[ply]); ply += 1
        lines.append(f"| {move_num} | {w} | {b} |")
        move_num += 1
    if len(lines) == 2:
        lines.append("| 1 |  |  |")
    return "\n".join(lines)


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="ChatGPT Chess (no images)", page_icon="♟️", layout="wide")
st.title("♟️ ChatGPT Chess (Streamlit, no images)")

with st.sidebar:
    st.subheader("Settings")
    default_color = st.session_state.get("assistant_color", "black")
    assistant_color = st.selectbox("Assistant plays:", ["white", "black"], index=(0 if default_color == "white" else 1))
    st.session_state["assistant_color"] = assistant_color

    model = st.text_input("OpenAI model", value=st.session_state.get("model", "gpt-4o-mini"))
    st.session_state["model"] = model

    api_key = st.text_input("OPENAI_API_KEY (optional)", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    auto_move = st.checkbox("Auto assistant move", value=st.session_state.get("auto", True))
    flip = st.checkbox("Flip board (show Black at bottom)", value=st.session_state.get("flip", False))
    show_legal = st.checkbox("Show legal moves (SAN)", value=st.session_state.get("show_legal", False))
    rng_seed = st.number_input("Retry seed (unused, kept for stability)", value=st.session_state.get("seed", 0), step=1)

    colb1, colb2, colb3 = st.columns(3)
    new_game = colb1.button("New Game")
    undo_move = colb2.button("Undo")
    assistant_now = colb3.button("Assistant Move Now")

# Init state
if new_game or "board" not in st.session_state:
    st.session_state.board = chess.Board()
    st.session_state.start_fen = st.session_state.board.fen()
    st.session_state.pending_engine = False
    st.session_state.seed = int(rng_seed)
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

board: chess.Board = st.session_state.board
start_fen: str = st.session_state.start_fen
assistant_is_white = (assistant_color == "white")

# Undo logic
if undo_move:
    if board.move_stack:
        board.pop()
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
    else:
        st.info("Nothing to undo.")

# Human move (only when it's the human's turn)
if not (assistant_is_white == board.turn) and not board.is_game_over():
    with st.form("human_move_form", clear_on_submit=True):
        mv_text = st.text_input("Your move (e.g., 'b2 to b4', 'e2e4', 'Nf3', 'O-O'):", key="mv_input")
        submitted = st.form_submit_button("Play move")
    if submitted:
        mv = parse_human_move(mv_text, board)
        if not mv:
            st.error("Couldn't parse or illegal. Try 'b2 to b4', 'e2e4', 'Nf3', or 'O-O'.")
        else:
            board.push(mv)
            st.session_state.pending_engine = True  # queue assistant move
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

# Assistant move trigger
client = create_client(api_key)
need_assistant = (assistant_is_white == board.turn) and not board.is_game_over()
if assistant_now:
    st.session_state.pending_engine = True
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()
if auto_move and need_assistant:
    st.session_state.pending_engine = True

# Assistant makes exactly one move when pending (NO heuristic; retries instead)
if st.session_state.pending_engine and need_assistant:
    st.session_state.pending_engine = False

    MAX_TRIES = 5
    last_err = None
    for attempt in range(1, MAX_TRIES + 1):
        try:
            if not client:
                raise RuntimeError("OPENAI_API_KEY not set.")
            mv, _ = assistant_move_with_model(client, board, model=model)
            board.push(mv)
            # Successful move → refresh UI immediately
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
        except Exception as e:
            last_err = e
            continue
        break  # executed only if no exception

    # If we couldn't get a legal move after retries, keep it the assistant's turn
    if last_err and (len(board.move_stack) == 0 or (assistant_is_white == board.turn)):
        st.session_state.pending_engine = True  # remain pending for next manual retry
        st.error(
            f"Assistant couldn't produce a legal move after {MAX_TRIES} tries. "
            f"It's still the assistant's turn. You can click **Assistant Move Now** "
            f"or change the model/API key and try again.\n\nLast error: {last_err}"
        )

# --------- RENDER AFTER ALL MOVE LOGIC ----------
svg = chess.svg.board(
    board=board,
    coordinates=True,
    lastmove=board.move_stack[-1] if board.move_stack else None,
    check=board.king(board.turn) if board.is_check() else None,
    orientation=(chess.BLACK if flip else chess.WHITE),
)
st.components.v1.html(svg, height=520, scrolling=False)

turn_label = "White" if board.turn else "Black"
st.write(f"**Turn:** {turn_label}  **FEN:** `{board.fen()}`")

if show_legal:
    st.caption("Legal moves:")
    st.write(", ".join(board.san(m) for m in board.legal_moves))

# Game end notice
if board.is_game_over():
    show_game_end(board)

# Moves table (rebuilt from the stack so it's always correct)
def moves_table_markdown(board: chess.Board, start_fen: str) -> str:
    tmp = chess.Board(start_fen)
    moves = list(board.move_stack)
    lines = ["| # | White | Black |", "|---:|:-----|:------|"]
    ply = 0
    move_num = 1
    while ply < len(moves):
        w = tmp.san(moves[ply]); tmp.push(moves[ply]); ply += 1
        b = ""
        if ply < len(moves):
            b = tmp.san(moves[ply]); tmp.push(moves[ply]); ply += 1
        lines.append(f"| {move_num} | {w} | {b} |")
        move_num += 1
    if len(lines) == 2:
        lines.append("| 1 |  |  |")
    return "\n".join(lines)

st.subheader("Moves")
st.markdown(moves_table_markdown(board, start_fen))

with st.expander("Text board (unicode)"):
    st.text(unicode_board(board))
# --- PGN download ---
import chess.pgn, io
pgn_game = chess.pgn.Game()
pgn_game.setup(chess.Board(st.session_state.start_fen))
node = pgn_game
tmp = chess.Board(st.session_state.start_fen)
for mv in board.move_stack:
    node = node.add_variation(mv)
    tmp.push(mv)
buf = io.StringIO()
print(pgn_game, file=buf)
st.download_button("Download PGN", buf.getvalue(), file_name="game.pgn")
