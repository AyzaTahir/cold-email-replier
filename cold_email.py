"""
Cold Email Replier — Streamlit + CrewAI
Generate personalized cold emails and optional auto-reply via SMTP.
"""

from __future__ import annotations

import os
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# CrewAI injects cache_breakpoint for Anthropic caching; Groq rejects it.
# https://github.com/crewAIInc/crewAI/issues/5886
import crewai.llms.cache as _crewai_cache

_crewai_cache.mark_cache_breakpoint = lambda msg: msg

import litellm

_UNSUPPORTED_LLM_MSG_KEYS = frozenset(
    {"cache_breakpoint", "is_litellm", "provider_specific_fields"}
)
_original_litellm_completion = litellm.completion


def _strip_unsupported_message_keys(kwargs: dict) -> dict:
    messages = kwargs.get("messages")
    if not messages:
        return kwargs
    cleaned = []
    for msg in messages:
        if isinstance(msg, dict):
            cleaned.append(
                {k: v for k, v in msg.items() if k not in _UNSUPPORTED_LLM_MSG_KEYS}
            )
        else:
            cleaned.append(msg)
    return {**kwargs, "messages": cleaned}


def _patched_litellm_completion(*args, **kwargs):
    kwargs = _strip_unsupported_message_keys(kwargs)
    return _original_litellm_completion(*args, **kwargs)


litellm.completion = _patched_litellm_completion

import streamlit as st
from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cold Email Replier",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "generated_email": "",
    "generated_reply": "",
    "last_action": "",
    "send_status": "",
}
for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_llm(api_key: str, model: str) -> LLM:
    if not api_key or not api_key.strip():
        raise ValueError("API key is required. Paste your Groq API key in the sidebar.")
    return LLM(model=model, api_key=api_key.strip())


def parse_email_output(raw: str) -> dict[str, str]:
    """Extract subject and body from agent output."""
    text = str(raw).strip()
    subject = ""
    body = text

    subject_match = re.search(
        r"(?i)^(?:subject|subject line)\s*:\s*(.+)$", text, re.MULTILINE
    )
    if subject_match:
        subject = subject_match.group(1).strip()
        body = re.sub(
            r"(?i)^(?:subject|subject line)\s*:\s*.+\n?",
            "",
            text,
            count=1,
            flags=re.MULTILINE,
        ).strip()

    return {"subject": subject or "Introduction — Opportunity to Connect", "body": body}


def send_email_smtp(
    *,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    sender_email: str,
    recipient_email: str,
    subject: str,
    body: str,
    use_tls: bool = True,
) -> tuple[bool, str]:
    if not all([smtp_host, smtp_user, smtp_password, sender_email, recipient_email]):
        return False, "Fill in all SMTP fields in the sidebar before sending."

    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            if use_tls:
                server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(sender_email, [recipient_email], msg.as_string())

        return True, f"Email sent to {recipient_email}."
    except Exception as exc:
        return False, f"Failed to send email: {exc}"


def run_cold_email_crew(
    llm: LLM,
    *,
    company_name: str,
    company_email: str,
    contact_person: str,
    industry: str,
    company_website: str,
    your_name: str,
    your_role: str,
    your_value_prop: str,
    email_goal: str,
    tone: str,
) -> str:
    writer = Agent(
        role="Cold Email Copywriter",
        goal=(
            "Write concise, personalized cold emails that get replies "
            "without sounding generic or salesy."
        ),
        backstory=(
            "You are a senior B2B outreach specialist with 10+ years of experience "
            "writing emails that open doors at startups and enterprises."
        ),
        llm=llm,
        verbose=False,
    )

    task = Task(
        description=f"""
Write a cold outreach email with these details:

**Recipient company:** {company_name}
**Recipient email:** {company_email}
**Contact person:** {contact_person or "Hiring manager / relevant decision maker"}
**Industry:** {industry or "Not specified"}
**Company website:** {company_website or "Not specified"}

**Sender name:** {your_name}
**Sender role:** {your_role}
**Value proposition:** {your_value_prop}
**Goal of this email:** {email_goal}
**Tone:** {tone}

Requirements:
- Start with Subject: on the first line, then a blank line, then the email body.
- Keep the body under 180 words.
- Personalize for {company_name}; avoid clichés like "I hope this finds you well".
- One clear call to action.
- Sign off with {your_name}.
""",
        expected_output=(
            "A complete email starting with 'Subject: ...' followed by the email body."
        ),
        agent=writer,
    )

    crew = Crew(
        agents=[writer],
        tasks=[task],
        process=Process.sequential,
        llm=llm,
        verbose=False,
        tracing=False,
    )
    return str(crew.kickoff())


def run_reply_crew(
    llm: LLM,
    *,
    incoming_email: str,
    company_name: str,
    your_name: str,
    your_role: str,
    reply_goal: str,
    tone: str,
) -> str:
    replier = Agent(
        role="Email Reply Specialist",
        goal="Draft professional, context-aware email replies that move conversations forward.",
        backstory=(
            "You excel at reading between the lines of business emails and crafting "
            "responses that are warm, direct, and action-oriented."
        ),
        llm=llm,
        verbose=False,
    )

    task = Task(
        description=f"""
Draft a reply to this incoming email:

---
{incoming_email}
---

**Company:** {company_name}
**Your name:** {your_name}
**Your role:** {your_role}
**Goal of your reply:** {reply_goal}
**Tone:** {tone}

Requirements:
- Start with Subject: on the first line (use Re: when appropriate), then a blank line, then the reply body.
- Address their points directly; keep under 150 words.
- End with a clear next step.
- Sign off with {your_name}.
""",
        expected_output=(
            "A complete reply starting with 'Subject: ...' followed by the reply body."
        ),
        agent=replier,
    )

    crew = Crew(
        agents=[replier],
        tasks=[task],
        process=Process.sequential,
        llm=llm,
        verbose=False,
        tracing=False,
    )
    return str(crew.kickoff())


# ---------------------------------------------------------------------------
# Sidebar — API keys & company config
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Dashboard")
    st.caption("Configure API keys and company details in real time.")

    st.subheader("🔑 API Configuration")
    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Paste your key here — it applies immediately on the next generate/send action.",
        key="groq_api_key_input",
    )
    model_choice = st.selectbox(
        "Model",
        [
            "groq/llama-3.3-70b-versatile",
            "groq/llama-3.1-8b-instant",
            "groq/mixtral-8x7b-32768",
        ],
        index=0,
    )

    st.divider()
    st.subheader("🏢 Target Company")
    company_name = st.text_input("Company name", placeholder="Acme Corp")
    company_email = st.text_input("Company email", placeholder="hr@acme.com")
    contact_person = st.text_input("Contact person", placeholder="Jane Doe")
    industry = st.text_input("Industry", placeholder="SaaS / Fintech / etc.")
    company_website = st.text_input("Website", placeholder="https://acme.com")

    st.divider()
    st.subheader("👤 Your Profile")
    your_name = st.text_input("Your name", placeholder="Alex Smith")
    your_role = st.text_input("Your role / title", placeholder="Software Engineer")
    your_value_prop = st.text_area(
        "Your value proposition",
        placeholder="Briefly describe what you offer or why you're reaching out.",
        height=90,
    )

    st.divider()
    st.subheader("📤 SMTP (auto-send)")
    st.caption("Optional — required only if you want to send emails from the app.")
    smtp_host = st.text_input("SMTP host", value="smtp.gmail.com")
    smtp_port = st.number_input("SMTP port", min_value=1, max_value=65535, value=587)
    smtp_user = st.text_input("SMTP username", placeholder="you@gmail.com")
    smtp_password = st.text_input("SMTP password / app password", type="password")
    sender_email = st.text_input("From email", placeholder="you@gmail.com")
    use_tls = st.checkbox("Use TLS", value=True)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("✉️ Cold Email Replier")
st.markdown(
    "Generate personalized cold emails with **CrewAI**, preview them, "
    "and optionally send or auto-reply to the company from the sidebar."
)

tab_compose, tab_reply = st.tabs(["📝 Compose Cold Email", "↩️ Reply to Email"])

# --- Compose tab ---
with tab_compose:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        email_goal = st.text_area(
            "Email goal",
            placeholder="e.g. Request a 15-minute intro call about a backend role",
            height=80,
        )
        tone = st.selectbox(
            "Tone",
            ["Professional", "Friendly", "Confident", "Concise"],
            key="compose_tone",
        )

        generate_btn = st.button("🚀 Generate Cold Email", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Generated email")
        if st.session_state.generated_email:
            parsed = parse_email_output(st.session_state.generated_email)
            st.text_input("Subject", value=parsed["subject"], key="compose_subject_display", disabled=True)
            st.text_area("Body", value=parsed["body"], height=280, key="compose_body_display")
        else:
            st.info("Fill in the sidebar and click **Generate Cold Email**.")

    if generate_btn:
        if not company_name or not company_email:
            st.error("Company name and company email are required in the sidebar.")
        elif not your_name:
            st.error("Your name is required in the sidebar.")
        else:
            with st.spinner("CrewAI is writing your cold email..."):
                try:
                    llm = build_llm(groq_api_key, model_choice)
                    raw = run_cold_email_crew(
                        llm,
                        company_name=company_name,
                        company_email=company_email,
                        contact_person=contact_person,
                        industry=industry,
                        company_website=company_website,
                        your_name=your_name,
                        your_role=your_role,
                        your_value_prop=your_value_prop,
                        email_goal=email_goal or "Introduce myself and start a conversation",
                        tone=tone,
                    )
                    st.session_state.generated_email = raw
                    st.session_state.last_action = "compose"
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    if st.session_state.generated_email:
        st.divider()
        send_col1, send_col2 = st.columns(2)
        with send_col1:
            auto_send = st.button("📨 Send to Company Email", use_container_width=True)
        with send_col2:
            if st.session_state.send_status:
                st.caption(st.session_state.send_status)

        if auto_send:
            parsed = parse_email_output(st.session_state.generated_email)
            ok, msg = send_email_smtp(
                smtp_host=smtp_host,
                smtp_port=int(smtp_port),
                smtp_user=smtp_user,
                smtp_password=smtp_password,
                sender_email=sender_email or smtp_user,
                recipient_email=company_email,
                subject=parsed["subject"],
                body=parsed["body"],
                use_tls=use_tls,
            )
            st.session_state.send_status = msg
            if ok:
                st.success(msg)
            else:
                st.error(msg)

# --- Reply tab ---
with tab_reply:
    col_a, col_b = st.columns([1, 1])

    with col_a:
        incoming_email = st.text_area(
            "Paste the email you received",
            placeholder="Paste the full email thread or message here...",
            height=220,
        )
        reply_goal = st.text_area(
            "Reply goal",
            placeholder="e.g. Confirm interest and propose times for a call",
            height=70,
        )
        reply_tone = st.selectbox(
            "Tone",
            ["Professional", "Friendly", "Confident", "Concise"],
            key="reply_tone",
        )
        reply_btn = st.button("🤖 Generate Reply", type="primary", use_container_width=True)
        auto_reply_send = st.checkbox("Auto-send reply after generation", value=False)

    with col_b:
        st.subheader("Generated reply")
        if st.session_state.generated_reply:
            parsed_reply = parse_email_output(st.session_state.generated_reply)
            st.text_input("Subject", value=parsed_reply["subject"], key="reply_subject_display", disabled=True)
            st.text_area("Body", value=parsed_reply["body"], height=280, key="reply_body_display")
        else:
            st.info("Paste an incoming email and click **Generate Reply**.")

    if reply_btn:
        if not incoming_email.strip():
            st.error("Paste the incoming email first.")
        elif not company_name or not company_email:
            st.error("Company name and email are required in the sidebar.")
        elif not your_name:
            st.error("Your name is required in the sidebar.")
        else:
            with st.spinner("CrewAI is drafting your reply..."):
                try:
                    llm = build_llm(groq_api_key, model_choice)
                    raw = run_reply_crew(
                        llm,
                        incoming_email=incoming_email,
                        company_name=company_name,
                        your_name=your_name,
                        your_role=your_role,
                        reply_goal=reply_goal or "Move the conversation forward professionally",
                        tone=reply_tone,
                    )
                    st.session_state.generated_reply = raw
                    st.session_state.last_action = "reply"

                    if auto_reply_send:
                        parsed_reply = parse_email_output(raw)
                        ok, msg = send_email_smtp(
                            smtp_host=smtp_host,
                            smtp_port=int(smtp_port),
                            smtp_user=smtp_user,
                            smtp_password=smtp_password,
                            sender_email=sender_email or smtp_user,
                            recipient_email=company_email,
                            subject=parsed_reply["subject"],
                            body=parsed_reply["body"],
                            use_tls=use_tls,
                        )
                        st.session_state.send_status = msg
                        if ok:
                            st.success(f"Reply generated and sent: {msg}")
                        else:
                            st.error(msg)
                    else:
                        st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    if st.session_state.generated_reply:
        if st.button("📨 Send Reply to Company", key="send_reply_btn", use_container_width=True):
            parsed_reply = parse_email_output(st.session_state.generated_reply)
            ok, msg = send_email_smtp(
                smtp_host=smtp_host,
                smtp_port=int(smtp_port),
                smtp_user=smtp_user,
                smtp_password=smtp_password,
                sender_email=sender_email or smtp_user,
                recipient_email=company_email,
                subject=parsed_reply["subject"],
                body=parsed_reply["body"],
                use_tls=use_tls,
            )
            if ok:
                st.success(msg)
            else:
                st.error(msg)
