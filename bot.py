import os
import io
import json
import time
import urllib.request
import urllib.parse
from dotenv import load_dotenv
from google import genai
from google.genai import types
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv()


def load_cr():
    with open("rules/cr.txt", "r", encoding="utf-8-sig") as f:
        return f.read()


CR_TEXT = load_cr()

SYSTEM_PROMPT = (
    "Você é o JudgeBot, um Juiz Oficial de Magic: The Gathering (Level 2+) "
    "especializado em Commander/EDH.\n\n"
    "VOCÊ RESPONDE TRÊS TIPOS DE PERGUNTA:\n"
    "1. Interações entre cartas — o que acontece quando X encontra Y\n"
    "2. Explicação de cartas — como funciona a mecânica de uma carta específica\n"
    "3. Perguntas de timing e regras gerais — posso fazer X em tal momento?\n\n"
    "SUAS REGRAS:\n"
    "- Responda SOMENTE perguntas relacionadas a Magic: The Gathering\n"
    "- Use as Comprehensive Rules (CR) como base para todas as regras do jogo\n"
    "- Quando o texto Oracle de uma carta for fornecido na mensagem (via Scryfall), "
    "use SEMPRE esse texto oficial — nunca sua memória sobre a carta\n"
    "- SEMPRE cite o número exato da regra aplicada (exemplo: CR 702.6d)\n"
    "- Se não tiver certeza ou a regra não estiver clara, diga explicitamente\n"
    "- Recuse educadamente qualquer pergunta não relacionada a Magic\n"
    "- Responda sempre em português brasileiro\n"
    "- Seja didático mas direto, como um juiz experiente explicando numa mesa\n\n"
    "FORMATO DE RESPOSTA:\n"
    "- Resposta clara e direta\n"
    "- Regra(s) aplicada(s): CR XXX.X (sempre que relevante)\n"
    "- Se houver exceções ou casos especiais importantes, mencione\n\n"
    "=== COMPREHENSIVE RULES OFICIAIS (Fevereiro/2026) ===\n\n"
    + CR_TEXT
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chat_histories: dict = {}
_last_scryfall_request: float = 0


def fetch_card(card_name: str) -> str | None:
    """Fetch Oracle text and rulings from Scryfall for a given card name."""
    global _last_scryfall_request

    elapsed = time.time() - _last_scryfall_request
    if elapsed < 0.1:
        time.sleep(0.1 - elapsed)

    try:
        card_name = card_name.strip()
        params = urllib.parse.urlencode({"fuzzy": card_name})
        url = f"https://api.scryfall.com/cards/named?{params}"
        headers = {"User-Agent": "MTGJudgeBot/1.0", "Accept": "application/json"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            card = json.loads(resp.read())
        _last_scryfall_request = time.time()

        # Handle double-faced cards
        oracle_text = card.get("oracle_text", "")
        if not oracle_text and "card_faces" in card:
            oracle_text = "\n//\n".join(
                f"{face.get('name', '')}: {face.get('oracle_text', '')}"
                for face in card["card_faces"]
            )

        # Fetch official rulings
        rulings_url = f"https://api.scryfall.com/cards/{card['id']}/rulings"
        req2 = urllib.request.Request(rulings_url, headers=headers)
        with urllib.request.urlopen(req2, timeout=5) as resp2:
            rulings = json.loads(resp2.read()).get("data", [])
        _last_scryfall_request = time.time()

        parts = [
            f"CARTA: {card['name']}",
            f"Custo: {card.get('mana_cost', 'N/A')}",
            f"Tipo: {card.get('type_line', 'N/A')}",
            f"Texto Oracle: {oracle_text or 'N/A'}",
        ]
        if card.get("power"):
            parts.append(f"Força/Resistência: {card['power']}/{card['toughness']}")
        if card.get("loyalty"):
            parts.append(f"Lealdade inicial: {card['loyalty']}")
        if card.get("keywords"):
            parts.append(f"Keywords: {', '.join(card['keywords'])}")
        if rulings:
            recent = [r["comment"] for r in rulings[-3:]]
            parts.append("Rulings oficiais:\n" + "\n".join(f"- {r}" for r in recent))

        return "\n".join(parts)

    except Exception as e:
        print(f"Scryfall erro para '{card_name}': {e}")
        return None


def extract_card_names(message: str) -> list[str]:
    """Ask Gemini to identify Magic card names in the user's message."""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=(
                "Identifique todos os nomes de cartas de Magic: The Gathering mencionados no texto abaixo. "
                "Retorne APENAS um array JSON com os nomes das cartas em inglês. "
                "Se nenhuma carta for mencionada, retorne []. Não inclua explicações.\n\n"
                f"Texto: {message}"
            ),
            config=types.GenerateContentConfig(temperature=0),
        )
        text = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"Erro ao extrair nomes de cartas: {e}")
        return []


def get_response(user_id: int, user_message: str) -> str:
    history = chat_histories.get(user_id, [])

    # Try to fetch card data from Scryfall
    card_names = extract_card_names(user_message)
    card_context = ""
    if card_names:
        fetched = []
        for name in card_names[:3]:  # Max 3 cards per query
            data = fetch_card(name)
            if data:
                fetched.append(data)
        if fetched:
            card_context = (
                "\n\n=== DADOS OFICIAIS DAS CARTAS (Scryfall) ===\n\n"
                + "\n\n---\n\n".join(fetched)
            )

    # Include card data in this turn's message, but store only the original in history
    message_for_model = user_message + card_context
    history.append(types.Content(role="user", parts=[types.Part(text=message_for_model)]))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
        ),
    )

    reply = response.text

    # Replace the enriched message with the original to keep history lean
    history[-1] = types.Content(role="user", parts=[types.Part(text=user_message)])
    history.append(types.Content(role="model", parts=[types.Part(text=reply)]))
    chat_histories[user_id] = history[-10:]

    return reply


def transcribe_audio(audio_bytes: bytes) -> str:
    """Send audio to Gemini and return the transcription."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=types.Content(
            role="user",
            parts=[
                types.Part(
                    inline_data=types.Blob(mime_type="audio/ogg", data=audio_bytes)
                ),
                types.Part(
                    text="Transcreva exatamente o que foi dito neste áudio. Retorne apenas a transcrição, sem comentários adicionais."
                ),
            ],
        ),
        config=types.GenerateContentConfig(temperature=0),
    )
    return response.text.strip()


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        voice_file = await context.bot.get_file(update.message.voice.file_id)
        buf = io.BytesIO()
        await voice_file.download_to_memory(buf)
        audio_bytes = buf.getvalue()

        transcribed = transcribe_audio(audio_bytes)
        if not transcribed:
            await update.message.reply_text("Não consegui entender o áudio. Tente novamente.")
            return

        reply = get_response(user_id, transcribed)
        await update.message.reply_text(
            f"🎙️ _{transcribed}_\n\n{reply}",
            parse_mode="Markdown",
        )
    except Exception as e:
        print(f"Erro ao processar áudio do usuário {user_id}: {e}")
        await update.message.reply_text(
            "Ocorreu um erro ao processar o áudio. Tente novamente."
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "⚖️ *JudgeBot — Juiz de Magic: The Gathering*\n\n"
        "Olá! Sou especialista em regras oficiais, com foco em Commander/EDH.\n\n"
        "Pode me perguntar sobre:\n"
        "• Stack e prioridade\n"
        "• Interações entre cartas específicas\n"
        "• State-based actions\n"
        "• Layers e efeitos de substituição\n"
        "• Regras específicas do Commander\n\n"
        "_Consulto o texto Oracle oficial das cartas via Scryfall em tempo real._\n"
        "_Aceito perguntas em texto ou áudio_ 🎙️\n"
        "_Use /reset para limpar o histórico da conversa._",
        parse_mode="Markdown",
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    chat_histories.pop(user_id, None)
    await update.message.reply_text("Histórico limpo! Pode fazer sua pergunta.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_message = update.message.text

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        reply = get_response(user_id, user_message)
        await update.message.reply_text(reply)
    except Exception as e:
        print(f"Erro ao processar mensagem do usuário {user_id}: {e}")
        chat_histories.pop(user_id, None)
        await update.message.reply_text(
            "Ocorreu um erro ao consultar as regras. Tente novamente."
        )


def main() -> None:
    telegram_token = os.getenv("TELEGRAM_TOKEN")
    if not telegram_token:
        raise ValueError("TELEGRAM_TOKEN não encontrado no arquivo .env")
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY não encontrado no arquivo .env")

    app = Application.builder().token(telegram_token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    webhook_base = os.getenv("WEBHOOK_URL")
    port = int(os.getenv("PORT", 8080))

    if webhook_base:
        webhook_url = f"{webhook_base.rstrip('/')}/{telegram_token}"
        print(f"JudgeBot iniciando com webhook em {webhook_base}")
        app.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=telegram_token,
            webhook_url=webhook_url,
        )
    else:
        print("JudgeBot rodando localmente (polling)...")
        app.run_polling()


if __name__ == "__main__":
    main()
